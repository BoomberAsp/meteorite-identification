"""
Ensemble engine: hard voting, soft voting, hierarchical ensemble.
Takes multiple model prediction CSVs (id, label, prob_m) and produces
combined predictions + comparison analysis.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime


def load_preds(csv_path):
    """Load a model's predictions: returns {id: {"label": int, "prob": float}}."""
    df = pd.read_csv(csv_path)
    if "prob_m" not in df.columns:
        df["prob_m"] = df["label"].astype(float)
    return {row["id"]: {"label": int(row["label"]), "prob": float(row["prob_m"])}
            for _, row in df.iterrows()}


def hard_voting(model_preds, tie_break="highest_prob"):
    """
    Hard voting: each model votes 0 or 1, take majority.
    tie_break: "highest_prob" uses the model with highest probability;
               "label_1" always picks 1; "label_0" always picks 0.
    Returns dict {id: int label}.
    """
    ids = list(model_preds[0].keys())
    results = {}
    for img_id in ids:
        votes = [m[img_id]["label"] for m in model_preds]
        vote_1 = sum(votes)
        vote_0 = len(votes) - vote_1
        if vote_1 > vote_0:
            results[img_id] = 1
        elif vote_0 > vote_1:
            results[img_id] = 0
        else:
            if tie_break == "highest_prob":
                best_prob = max(m[img_id]["prob"] if m[img_id]["label"] == 1
                                else 1 - m[img_id]["prob"] for m in model_preds)
                # Pick the label of the model with highest confidence
                for m in model_preds:
                    confidence = m[img_id]["prob"] if m[img_id]["label"] == 1 else 1 - m[img_id]["prob"]
                    if confidence == best_prob:
                        results[img_id] = m[img_id]["label"]
                        break
            elif tie_break == "label_1":
                results[img_id] = 1
            else:
                results[img_id] = 0
    return results


def soft_voting(model_preds, weights=None, threshold=0.5):
    """
    Soft voting: weighted average of probabilities.
    Returns dict {id: {"prob": float, "label": int}}.
    """
    ids = list(model_preds[0].keys())
    n = len(model_preds)
    if weights is None:
        weights = [1.0 / n] * n
    weights = np.array(weights) / np.sum(weights)

    results = {}
    for img_id in ids:
        avg_prob = sum(w * m[img_id]["prob"] for w, m in zip(weights, model_preds))
        results[img_id] = {"prob": avg_prob, "label": 1 if avg_prob >= threshold else 0}

    results["_threshold"] = threshold
    return results


def hierarchical_ensemble(primary_preds, secondary_preds_list, low=0.3, high=0.7):
    """
    Hierarchical: primary model decides for high-confidence samples;
    secondary models (hard-vote) decide for uncertain samples.
    Returns dict {id: {"prob": float, "label": int, "source": str}}.
    """
    ids = list(primary_preds.keys())
    results = {}
    primary_decided = 0
    secondary_decided = 0

    for img_id in ids:
        prob = primary_preds[img_id]["prob"]
        if prob >= high or prob <= low:
            results[img_id] = {"prob": prob, "label": 1 if prob >= 0.5 else 0,
                               "source": "primary"}
            primary_decided += 1
        else:
            # Secondary models vote
            votes = [m[img_id]["label"] for m in secondary_preds_list]
            vote_1 = sum(votes)
            vote_0 = len(votes) - vote_1
            avg_secondary_prob = np.mean([m[img_id]["prob"] for m in secondary_preds_list])
            results[img_id] = {"prob": avg_secondary_prob,
                               "label": 1 if vote_1 >= vote_0 else 0,
                               "source": "secondary"}
            secondary_decided += 1

    results["_stats"] = {"primary_decided": primary_decided, "secondary_decided": secondary_decided}
    return results


def compare_predictions(preds_list, names):
    """
    Compare predictions across models: disagreement rate, agreement matrix.
    """
    ids = list(preds_list[0].keys())
    n = len(preds_list)

    # Pairwise disagreement
    comparisons = {}
    for i in range(n):
        for j in range(i + 1, n):
            disagree = sum(
                1 for img_id in ids
                if preds_list[i][img_id]["label"] != preds_list[j][img_id]["label"]
            )
            rate = disagree / len(ids)
            comparisons[f"{names[i]} vs {names[j]}"] = {
                "disagree_count": disagree,
                "disagree_rate": round(rate, 4),
                "total": len(ids),
            }

    return comparisons


def make_submission(results, template_csv_path, output_path):
    """Generate Kaggle submission CSV from ensemble results."""
    template = pd.read_csv(template_csv_path)
    label_map = {}
    for img_id, info in results.items():
        if isinstance(info, dict) and "label" in info:
            label_map[img_id] = info["label"]
        elif isinstance(info, (int, np.integer)):
            label_map[img_id] = int(info)

    template["label"] = template["id"].map(label_map).astype(int)
    if template["label"].isna().any():
        missing = template.loc[template["label"].isna(), "id"].head(5).tolist()
        raise RuntimeError(f"Missing predictions: {missing}")
    template.to_csv(output_path, index=False)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Model ensemble for meteorite identification")
    parser.add_argument("--pred_files", nargs="+", required=True,
                        help="List of prediction CSV files")
    parser.add_argument("--names", nargs="+", required=True,
                        help="Short names for each model (same order as --pred_files)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for ensemble results")
    parser.add_argument("--template_csv", required=True,
                        help="Path to sample_submission.csv")
    parser.add_argument("--primary_idx", type=int, default=0,
                        help="Index of primary model in --pred_files (for hierarchical)")
    parser.add_argument("--secondary_indices", nargs="+", type=int, default=None,
                        help="Indices of secondary models (for hierarchical)")
    parser.add_argument("--soft_weights", nargs="+", type=float, default=None,
                        help="Weights for soft voting")
    parser.add_argument("--hierarchical_low", type=float, default=0.3,
                        help="Low threshold for hierarchical ensemble")
    parser.add_argument("--hierarchical_high", type=float, default=0.7,
                        help="High threshold for hierarchical ensemble")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for soft voting & hard label generation (default: 0.5)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load all predictions
    all_preds = [load_preds(f) for f in args.pred_files]
    print(f"Loaded {len(all_preds)} models: {args.names}")

    # Verify same IDs
    ids_list = [set(p.keys()) for p in all_preds]
    common_ids = ids_list[0]
    for i, ids in enumerate(ids_list):
        common_ids = common_ids & ids
    if len(common_ids) == 0:
        raise RuntimeError("No common IDs across models!")
    print(f"Common test images: {len(common_ids)}")

    report = {"timestamp": timestamp, "models": args.names, "num_images": len(common_ids)}
    report["model_disagreement"] = compare_predictions(all_preds, args.names)

    # ── Strategy 1: Hard Voting ──
    print("\n=== Hard Voting ===")
    hard_results = hard_voting(all_preds)
    hard_submission = os.path.join(args.output_dir, f"ensemble_hard_{timestamp}.csv")
    make_submission(hard_results, args.template_csv, hard_submission)
    n_pos_hard = sum(1 for v in hard_results.values() if v == 1)
    report["hard_voting"] = {"submission": hard_submission, "pred_pos_count": n_pos_hard}
    print(f"  Submission: {hard_submission}")
    print(f"  Meteorite predictions: {n_pos_hard}")

    # ── Strategy 2: Soft Voting ──
    print("\n=== Soft Voting ===")
    soft_results = soft_voting(all_preds, weights=args.soft_weights, threshold=args.threshold)
    soft_submission = os.path.join(args.output_dir, f"ensemble_soft_{timestamp}.csv")
    make_submission(soft_results, args.template_csv, soft_submission)
    n_pos_soft = sum(1 for v in soft_results.values()
                     if isinstance(v, dict) and v.get("label") == 1)
    threshold_used = soft_results.pop("_threshold", None) if "_threshold" in soft_results else "top101"
    report["soft_voting"] = {
        "submission": soft_submission,
        "pred_pos_count": n_pos_soft,
        "weights": args.soft_weights if args.soft_weights else "equal",
        "threshold": str(threshold_used),
    }
    print(f"  Submission: {soft_submission}")
    print(f"  Meteorite predictions: {n_pos_soft}")
    print(f"  Threshold: {threshold_used}")

    # ── Strategy 3: Hierarchical Ensemble ──
    print("\n=== Hierarchical Ensemble ===")
    primary = all_preds[args.primary_idx]
    if args.secondary_indices is None:
        secondary_indices = [i for i in range(len(all_preds)) if i != args.primary_idx]
    else:
        secondary_indices = args.secondary_indices
    secondary = [all_preds[i] for i in secondary_indices]
    secondary_names = [args.names[i] for i in secondary_indices]

    hier_results = hierarchical_ensemble(primary, secondary,
                                         low=args.hierarchical_low,
                                         high=args.hierarchical_high)
    hier_stats = hier_results.pop("_stats", {})
    hier_submission = os.path.join(args.output_dir, f"ensemble_hierarchical_{timestamp}.csv")
    make_submission(hier_results, args.template_csv, hier_submission)
    n_pos_hier = sum(1 for v in hier_results.values()
                     if isinstance(v, dict) and v.get("label") == 1)
    report["hierarchical"] = {
        "submission": hier_submission,
        "pred_pos_count": n_pos_hier,
        "primary_model": args.names[args.primary_idx],
        "secondary_models": secondary_names,
        "low_threshold": args.hierarchical_low,
        "high_threshold": args.hierarchical_high,
        "primary_decided": hier_stats.get("primary_decided", 0),
        "secondary_decided": hier_stats.get("secondary_decided", 0),
    }
    print(f"  Submission: {hier_submission}")
    print(f"  Primary decided: {hier_stats.get('primary_decided', 0)}")
    print(f"  Secondary decided: {hier_stats.get('secondary_decided', 0)}")

    # ── Homogeneous Control: Swin-B only ensemble ──
    # If we have ≥2 models of same type, run homogeneous ensemble
    print("\n=== Control: Homogeneous Ensemble ===")
    report["homogeneous_control"] = {}
    for i, name in enumerate(args.names):
        # Single model as-is (this is the baseline from the prediction file)
        single = all_preds[i]
        single_submission = os.path.join(args.output_dir, f"single_{name}_{timestamp}.csv")
        make_submission(single, args.template_csv, single_submission)
        n_pos = sum(1 for v in single.values() if v.get("label") == 1)
        report["homogeneous_control"][f"single_{name}"] = {
            "submission": single_submission,
            "pred_pos_count": n_pos,
        }

    # ── Save Report ──
    report_path = os.path.join(args.output_dir, f"ensemble_report_{timestamp}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {report_path}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Files generated:")
    for key in ["hard_voting", "soft_voting", "hierarchical"]:
        if key in report:
            print(f"  {key}: {report[key]['submission']}")
    for model_key in report.get("homogeneous_control", {}):
        print(f"  {model_key}: {report['homogeneous_control'][model_key]['submission']}")
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
