#!/usr/bin/env python3
"""
Meteorite Image Labeling Web Application — Flask Backend

Supports two datasets:
  - Test set:  data/test_images-v2        (194 images, labels in data/labels_v2.json)
  - Train set: data/train_images_extracted_white_bg (5098 images, labels from JSONL)

Usage:
    python server.py              # default: http://localhost:5000
    PORT=8080 python server.py    # custom port
"""

import csv
import io
import json
import os
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Test set
TEST_IMAGE_DIR = DATA_DIR / "test_images-v2"
TEST_LABELS_FILE = DATA_DIR / "labels_v2.json"

# Training set
TRAIN_IMAGE_DIR = DATA_DIR / "train_images_extracted_white_bg"
TRAIN_JSONL_FILE = DATA_DIR / "merged" / "training_data_merged.jsonl"
TRAIN_LABELS_FILE = DATA_DIR / "labels_train.json"

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers — Generic
# ═══════════════════════════════════════════════════════════════════════════════

def _list_images(image_dir: Path) -> list[str]:
    """Return sorted list of image filenames from a directory."""
    if not image_dir.exists():
        return []
    return sorted(
        f.name
        for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )


def _load_json_labels(labels_file: Path) -> dict[str, int]:
    """Load labels from a JSON file. Returns {filename: 0|1}."""
    if labels_file.exists():
        try:
            with open(labels_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: int(v) for k, v in data.items() if int(v) in (0, 1)}
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _save_json_labels(labels: dict[str, int], labels_file: Path) -> None:
    """Save labels dict to JSON file."""
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def _generate_thumbnail(image_path: Path, size: tuple[int, int] = (150, 150)) -> io.BytesIO:
    """Generate a thumbnail of the given image, returned as BytesIO buffer."""
    img = Image.open(image_path)
    img.thumbnail(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    buf.seek(0)
    return buf


def _make_progress(image_dir: Path, labels: dict) -> dict:
    """Build progress dict for a dataset."""
    images = _list_images(image_dir)
    labeled = sum(1 for img in images if img in labels)
    meteorite = sum(1 for img in images if labels.get(img) == 1)
    non_meteorite = sum(1 for img in images if labels.get(img) == 0)
    return {
        "labeled": labeled,
        "total": len(images),
        "meteorite": meteorite,
        "non_meteorite": non_meteorite,
    }


def _export_csv(image_dir: Path, labels: dict, default_label: int = 0) -> str:
    """Generate CSV string matching sample_submission format for a dataset."""
    images = _list_images(image_dir)
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    writer.writerow(["id", "label"])
    for img in images:
        writer.writerow([img, labels.get(img, default_label)])
    output.seek(0)
    return output.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers — Training Set JSONL
# ═══════════════════════════════════════════════════════════════════════════════

def load_train_jsonl() -> dict[str, dict]:
    """Parse training JSONL.

    Returns {filename: {"label": 0|1, "reasoning": str, "instruction": str}}
    Filename is derived from source_index (e.g. source_index=1 → "000001.jpg").
    """
    result: dict[str, dict] = {}
    if not TRAIN_JSONL_FILE.exists():
        return result
    with open(TRAIN_JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            source_index = item.get("source_index")
            if source_index is None:
                continue

            output = item.get("output", "")
            # Determine label from output (starts with "是" = meteorite, "否" = non-meteorite)
            if output.startswith("是"):
                label = 1
            elif output.startswith("否"):
                label = 0
            else:
                label = 0  # fallback

            # Extract reasoning (text after "是。" or "否。")
            reasoning = ""
            for prefix in ("是。", "否。"):
                if prefix in output:
                    reasoning = output.split(prefix, 1)[1].strip()
                    break

            filename = f"{source_index:06d}.jpg"
            result[filename] = {
                "label": label,
                "reasoning": reasoning,
                "instruction": item.get("instruction", ""),
            }
    return result


def load_train_labels() -> dict[str, int]:
    """Load human-override labels for training set."""
    return _load_json_labels(TRAIN_LABELS_FILE)


def save_train_labels(labels: dict[str, int]) -> None:
    """Save human-override labels for training set."""
    _save_json_labels(labels, TRAIN_LABELS_FILE)


# ═══════════════════════════════════════════════════════════════════════════════
#  Routes: Page
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the main labeling UI page."""
    return render_template("index.html")


# ═══════════════════════════════════════════════════════════════════════════════
#  Routes: Test Set API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/test/images")
def api_test_images():
    images = _list_images(TEST_IMAGE_DIR)
    return jsonify({"images": images, "total": len(images)})


@app.route("/api/test/image/<filename>")
def api_test_image(filename: str):
    image_path = TEST_IMAGE_DIR / filename
    if not image_path.exists():
        return jsonify({"error": "Image not found"}), 404
    return send_file(image_path, mimetype="image/jpeg")


@app.route("/api/test/thumbnail/<filename>")
def api_test_thumbnail(filename: str):
    image_path = TEST_IMAGE_DIR / filename
    if not image_path.exists():
        return jsonify({"error": "Image not found"}), 404
    thumb = _generate_thumbnail(image_path)
    return send_file(thumb, mimetype="image/jpeg")


@app.route("/api/test/labels", methods=["GET"])
def api_test_get_labels():
    labels = _load_json_labels(TEST_LABELS_FILE)
    return jsonify({"labels": labels})


@app.route("/api/test/label", methods=["POST"])
def api_test_set_label():
    """Set/clear the human label for a single test image."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    filename = data.get("filename", "")
    label = data.get("label")
    image_path = TEST_IMAGE_DIR / filename

    if not image_path.exists():
        return jsonify({"error": f"Image not found: {filename}"}), 404

    labels = _load_json_labels(TEST_LABELS_FILE)

    if label is None or label == -1:
        labels.pop(filename, None)
    else:
        label = int(label)
        if label not in (0, 1):
            return jsonify({"error": "Label must be 0 or 1"}), 400
        labels[filename] = label

    _save_json_labels(labels, TEST_LABELS_FILE)
    progress = _make_progress(TEST_IMAGE_DIR, labels)

    return jsonify({
        "status": "ok",
        "filename": filename,
        "label": labels.get(filename),
        "progress": progress,
    })


@app.route("/api/test/progress")
def api_test_progress():
    labels = _load_json_labels(TEST_LABELS_FILE)
    return jsonify(_make_progress(TEST_IMAGE_DIR, labels))


@app.route("/api/test/export")
def api_test_export():
    labels = _load_json_labels(TEST_LABELS_FILE)
    csv_text = _export_csv(TEST_IMAGE_DIR, labels, default_label=0)
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=submission_test_labels.csv",
            "Content-Type": "text/csv; charset=utf-8",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Routes: Training Set API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/training/images")
def api_training_images():
    images = _list_images(TRAIN_IMAGE_DIR)
    return jsonify({"images": images, "total": len(images)})


@app.route("/api/training/image/<filename>")
def api_training_image(filename: str):
    image_path = TRAIN_IMAGE_DIR / filename
    if not image_path.exists():
        return jsonify({"error": "Image not found"}), 404
    return send_file(image_path, mimetype="image/jpeg")


@app.route("/api/training/thumbnail/<filename>")
def api_training_thumbnail(filename: str):
    image_path = TRAIN_IMAGE_DIR / filename
    if not image_path.exists():
        return jsonify({"error": "Image not found"}), 404
    thumb = _generate_thumbnail(image_path)
    return send_file(thumb, mimetype="image/jpeg")


@app.route("/api/training/info/<filename>")
def api_training_info(filename: str):
    """Return JSONL info for a training image: original label, reasoning, instruction."""
    jsonl_data = load_train_jsonl()
    entry = jsonl_data.get(filename)
    if entry is None:
        return jsonify({
            "filename": filename,
            "original_label": None,
            "reasoning": "",
            "instruction": "",
            "in_jsonl": False,
        })
    return jsonify({
        "filename": filename,
        "original_label": entry["label"],
        "reasoning": entry["reasoning"],
        "instruction": entry["instruction"],
        "in_jsonl": True,
    })


@app.route("/api/training/labels", methods=["GET"])
def api_training_get_labels():
    """Get human-override labels for training set, plus JSONL base labels."""
    human_labels = load_train_labels()
    jsonl_data = load_train_jsonl()
    # Build base labels from JSONL
    base_labels = {fn: d["label"] for fn, d in jsonl_data.items()}
    return jsonify({
        "base_labels": base_labels,    # labels from JSONL (original)
        "human_labels": human_labels,  # human overrides
    })


@app.route("/api/training/label", methods=["POST"])
def api_training_set_label():
    """Set/clear human override label for a training image."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    filename = data.get("filename", "")
    label = data.get("label")
    image_path = TRAIN_IMAGE_DIR / filename

    if not image_path.exists():
        return jsonify({"error": f"Image not found: {filename}"}), 404

    labels = load_train_labels()

    if label is None or label == -1:
        labels.pop(filename, None)
    else:
        label = int(label)
        if label not in (0, 1):
            return jsonify({"error": "Label must be 0 or 1"}), 400
        labels[filename] = label

    save_train_labels(labels)
    progress = _make_progress(TRAIN_IMAGE_DIR, labels)

    return jsonify({
        "status": "ok",
        "filename": filename,
        "label": labels.get(filename),
        "progress": progress,
    })


@app.route("/api/training/progress")
def api_training_progress():
    labels = load_train_labels()
    return jsonify(_make_progress(TRAIN_IMAGE_DIR, labels))


@app.route("/api/training/export")
def api_training_export():
    """Export training labels as CSV.

    Uses human-override labels where available, falling back to JSONL base labels.
    Unlabeled (no JSONL, no human) defaults to 0.
    """
    human_labels = load_train_labels()
    jsonl_data = load_train_jsonl()

    # Merge: human overrides take priority over JSONL base
    merged = {}
    for fn, d in jsonl_data.items():
        merged[fn] = d["label"]  # JSONL base
    merged.update(human_labels)  # human overrides

    csv_text = _export_csv(TRAIN_IMAGE_DIR, merged, default_label=0)
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=submission_train_labels.csv",
            "Content-Type": "text/csv; charset=utf-8",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Routes: Legacy /api/* (for backward compatibility with old frontend)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/images")
def api_images_legacy():
    return api_test_images()


@app.route("/api/image/<filename>")
def api_image_legacy(filename: str):
    return api_test_image(filename)


@app.route("/api/thumbnail/<filename>")
def api_thumbnail_legacy(filename: str):
    return api_test_thumbnail(filename)


@app.route("/api/labels", methods=["GET"])
def api_get_labels_legacy():
    return api_test_get_labels()


@app.route("/api/label", methods=["POST"])
def api_set_label_legacy():
    return api_test_set_label()


@app.route("/api/progress")
def api_progress_legacy():
    return api_test_progress()


@app.route("/api/export")
def api_export_legacy():
    return api_test_export()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    test_count = len(_list_images(TEST_IMAGE_DIR))
    train_count = len(_list_images(TRAIN_IMAGE_DIR))
    jsonl_count = len(load_train_jsonl())

    print(f"\n  ☄  Meteorite Labeling Tool")
    print(f"  ───────────────────────────")
    print(f"  Test set:  {test_count} images  ({TEST_IMAGE_DIR})")
    print(f"  Train set: {train_count} images  ({TRAIN_IMAGE_DIR})")
    print(f"  JSONL:     {jsonl_count} entries ({TRAIN_JSONL_FILE})")
    print(f"  Server:    http://localhost:{port}")
    print(f"\n  Press Ctrl+C to stop.\n")

    app.run(host="0.0.0.0", port=port, debug=True)
