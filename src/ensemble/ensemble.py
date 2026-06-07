"""
Strict ensemble engine for meteorite identification.

Input prediction CSVs must contain:
    id,label[,prob_m]

Outputs:
    - Kaggle-style submission CSVs in the template order.
    - Per-strategy debug CSVs with probabilities, labels, and model votes.
    - One JSON report summarizing disagreements and positive counts.
"""

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"id", "label"}
PROB_COLUMN = "prob_m"


@dataclass(frozen=True)
class ModelPredictions:
    name: str
    path: str
    frame: pd.DataFrame


def _sample(values: Iterable[str], limit: int = 5) -> list[str]:
    return sorted(str(v) for v in values)[:limit]


def _require_binary_labels(df: pd.DataFrame, source: str) -> None:
    labels = set(df["label"].dropna().astype(int).tolist())
    invalid = labels - {0, 1}
    if invalid:
        raise ValueError(f"{source} has non-binary labels: {sorted(invalid)}")


def load_template(template_csv_path: str) -> pd.DataFrame:
    template = pd.read_csv(template_csv_path)
    if "id" not in template.columns:
        raise ValueError(f"{template_csv_path} must contain an 'id' column")
    if template["id"].duplicated().any():
        dupes = template.loc[template["id"].duplicated(), "id"].head(5).tolist()
        raise ValueError(f"{template_csv_path} contains duplicate ids: {dupes}")
    return template[["id"]].copy()


def load_prediction_file(csv_path: str, name: str) -> ModelPredictions:
    df = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing_columns)}")

    if df["id"].duplicated().any():
        dupes = df.loc[df["id"].duplicated(), "id"].head(5).tolist()
        raise ValueError(f"{csv_path} contains duplicate ids: {dupes}")

    if PROB_COLUMN not in df.columns:
        df[PROB_COLUMN] = df["label"].astype(float)

    df = df[["id", "label", PROB_COLUMN]].copy()
    df["label"] = df["label"].astype(int)
    df[PROB_COLUMN] = df[PROB_COLUMN].astype(float)

    _require_binary_labels(df, csv_path)

    bad_probs = df[(df[PROB_COLUMN] < 0.0) | (df[PROB_COLUMN] > 1.0)]
    if not bad_probs.empty:
        examples = bad_probs[["id", PROB_COLUMN]].head(5).to_dict("records")
        raise ValueError(f"{csv_path} has probabilities outside [0, 1]: {examples}")

    return ModelPredictions(name=name, path=csv_path, frame=df)


def validate_against_template(models: list[ModelPredictions], template: pd.DataFrame) -> None:
    template_ids = set(template["id"])
    for model in models:
        pred_ids = set(model.frame["id"])
        missing = template_ids - pred_ids
        extra = pred_ids - template_ids
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing {len(missing)} template ids, e.g. {_sample(missing)}")
            if extra:
                details.append(f"has {len(extra)} ids not in template, e.g. {_sample(extra)}")
            raise ValueError(f"Prediction file for '{model.name}' does not match template: {'; '.join(details)}")


def build_aligned_table(models: list[ModelPredictions], template: pd.DataFrame) -> pd.DataFrame:
    aligned = template.copy()
    for model in models:
        renamed = model.frame.rename(
            columns={
                "label": f"{model.name}_label",
                PROB_COLUMN: f"{model.name}_prob",
            }
        )
        aligned = aligned.merge(renamed, on="id", how="left", validate="one_to_one")

    missing = aligned.columns[aligned.isna().any()].tolist()
    if missing:
        raise RuntimeError(f"Aligned prediction table unexpectedly has missing values in: {missing}")
    return aligned


def normalize_weights(weights: list[float] | None, n_models: int) -> np.ndarray:
    if weights is None:
        return np.ones(n_models, dtype=float) / n_models
    if len(weights) != n_models:
        raise ValueError(f"Expected {n_models} weights, got {len(weights)}")
    arr = np.asarray(weights, dtype=float)
    if np.any(arr < 0):
        raise ValueError("Weights must be non-negative")
    if float(arr.sum()) <= 0:
        raise ValueError("At least one weight must be positive")
    return arr / arr.sum()


def _label_columns(names: list[str]) -> list[str]:
    return [f"{name}_label" for name in names]


def _prob_columns(names: list[str]) -> list[str]:
    return [f"{name}_prob" for name in names]


def single_model(aligned: pd.DataFrame, name: str) -> pd.DataFrame:
    result = aligned[["id", f"{name}_label", f"{name}_prob"]].copy()
    result = result.rename(columns={f"{name}_label": "label", f"{name}_prob": PROB_COLUMN})
    result["source"] = name
    return result


def soft_voting(
    aligned: pd.DataFrame,
    names: list[str],
    weights: list[float] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    norm_weights = normalize_weights(weights, len(names))
    probs = aligned[_prob_columns(names)].to_numpy(dtype=float)
    avg_prob = probs @ norm_weights

    result = aligned[["id"]].copy()
    result[PROB_COLUMN] = avg_prob
    result["label"] = (result[PROB_COLUMN] >= threshold).astype(int)
    result["source"] = "soft"
    return result


def hard_voting(
    aligned: pd.DataFrame,
    names: list[str],
    tie_break: str = "soft",
    threshold: float = 0.5,
) -> pd.DataFrame:
    if tie_break not in {"soft", "highest_confidence", "label_0", "label_1"}:
        raise ValueError("--hard_tie_break must be one of: soft, highest_confidence, label_0, label_1")

    labels = aligned[_label_columns(names)].to_numpy(dtype=int)
    probs = aligned[_prob_columns(names)].to_numpy(dtype=float)
    vote_count = labels.sum(axis=1)
    half = len(names) / 2.0
    out_labels = np.empty(len(aligned), dtype=int)
    sources = np.full(len(aligned), "hard_majority", dtype=object)

    majority_1 = vote_count > half
    majority_0 = vote_count < half
    tied = ~(majority_1 | majority_0)
    out_labels[majority_1] = 1
    out_labels[majority_0] = 0

    if tied.any():
        if tie_break == "label_1":
            out_labels[tied] = 1
            sources[tied] = "hard_tie_label_1"
        elif tie_break == "label_0":
            out_labels[tied] = 0
            sources[tied] = "hard_tie_label_0"
        elif tie_break == "soft":
            soft_probs = probs.mean(axis=1)
            out_labels[tied] = (soft_probs[tied] >= threshold).astype(int)
            sources[tied] = "hard_tie_soft"
        else:
            confidence = np.where(labels == 1, probs, 1.0 - probs)
            best_model_idx = confidence.argmax(axis=1)
            out_labels[tied] = labels[np.arange(len(labels)), best_model_idx][tied]
            sources[tied] = [
                f"hard_tie_{names[idx]}" for idx in best_model_idx[tied]
            ]

    result = aligned[["id"]].copy()
    result["label"] = out_labels
    result[PROB_COLUMN] = probs.mean(axis=1)
    result["vote_1"] = vote_count
    result["vote_0"] = len(names) - vote_count
    result["source"] = sources
    return result


def hierarchical_ensemble(
    aligned: pd.DataFrame,
    names: list[str],
    primary_idx: int = 0,
    secondary_indices: list[int] | None = None,
    low: float = 0.3,
    high: float = 0.7,
    threshold: float = 0.5,
    secondary_weights: list[float] | None = None,
) -> pd.DataFrame:
    if not 0 <= primary_idx < len(names):
        raise ValueError(f"--primary_idx out of range: {primary_idx}")
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("Hierarchical thresholds must satisfy 0 <= low <= high <= 1")

    if secondary_indices is None:
        secondary_indices = [i for i in range(len(names)) if i != primary_idx]
    if not secondary_indices:
        raise ValueError("Hierarchical ensemble needs at least one secondary model")
    if any(i < 0 or i >= len(names) for i in secondary_indices):
        raise ValueError(f"--secondary_indices out of range: {secondary_indices}")
    if primary_idx in secondary_indices:
        raise ValueError("--secondary_indices must not include --primary_idx")

    primary = names[primary_idx]
    secondary_names = [names[i] for i in secondary_indices]
    primary_prob = aligned[f"{primary}_prob"].to_numpy(dtype=float)
    primary_label = (primary_prob >= threshold).astype(int)

    confident = (primary_prob <= low) | (primary_prob >= high)
    secondary_result = soft_voting(
        aligned,
        secondary_names,
        weights=secondary_weights,
        threshold=threshold,
    )

    result = aligned[["id"]].copy()
    result[PROB_COLUMN] = np.where(confident, primary_prob, secondary_result[PROB_COLUMN].to_numpy())
    result["label"] = np.where(confident, primary_label, secondary_result["label"].to_numpy()).astype(int)
    result["source"] = np.where(confident, f"primary:{primary}", "secondary_soft")
    return result


def prediction_summary(result: pd.DataFrame) -> dict:
    total = int(len(result))
    positives = int(result["label"].sum())
    probs = result[PROB_COLUMN].astype(float)
    return {
        "num_images": total,
        "pred_pos_count": positives,
        "pred_neg_count": total - positives,
        "pos_rate": round(positives / total, 6) if total else 0.0,
        "prob_mean": float(probs.mean()),
        "prob_std": float(probs.std(ddof=0)),
        "prob_min": float(probs.min()),
        "prob_max": float(probs.max()),
    }


def disagreement_report(aligned: pd.DataFrame, names: list[str]) -> dict:
    report = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            disagree = (aligned[f"{left}_label"] != aligned[f"{right}_label"])
            report[f"{left} vs {right}"] = {
                "disagree_count": int(disagree.sum()),
                "disagree_rate": round(float(disagree.mean()), 6),
                "total": int(len(aligned)),
            }
    return report


def write_submission(result: pd.DataFrame, output_path: str) -> str:
    submission = result[["id", "label"]].copy()
    if submission["label"].isna().any():
        missing = submission.loc[submission["label"].isna(), "id"].head(5).tolist()
        raise RuntimeError(f"Missing predictions for ids: {missing}")
    submission["label"] = submission["label"].astype(int)
    submission.to_csv(output_path, index=False)
    return output_path


def write_debug(
    result: pd.DataFrame,
    aligned: pd.DataFrame,
    names: list[str],
    output_path: str,
) -> str:
    debug = result.copy()
    model_columns = []
    for name in names:
        model_columns.extend([f"{name}_label", f"{name}_prob"])
    debug = debug.merge(aligned[["id"] + model_columns], on="id", how="left", validate="one_to_one")
    debug.to_csv(output_path, index=False)
    return output_path


def save_strategy_outputs(
    result: pd.DataFrame,
    aligned: pd.DataFrame,
    names: list[str],
    output_dir: str,
    strategy_name: str,
    timestamp: str,
) -> dict:
    submission_path = os.path.join(output_dir, f"ensemble_{strategy_name}_{timestamp}.csv")
    debug_path = os.path.join(output_dir, f"ensemble_{strategy_name}_debug_{timestamp}.csv")
    write_submission(result, submission_path)
    write_debug(result, aligned, names, debug_path)

    summary = prediction_summary(result)
    summary.update({
        "submission": submission_path,
        "debug_csv": debug_path,
    })
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict ensemble for meteorite identification")
    parser.add_argument("--pred_files", nargs="+", required=True, help="Prediction CSV files")
    parser.add_argument("--names", nargs="+", required=True, help="Model names, same order as --pred_files")
    parser.add_argument("--template_csv", required=True, help="sample_submission.csv path")
    parser.add_argument("--output_dir", required=True, help="Directory for ensemble outputs")
    parser.add_argument("--strategies", nargs="+", default=["hard", "soft", "hierarchical"],
                        choices=["hard", "soft", "hierarchical", "single"],
                        help="Strategies to run")
    parser.add_argument("--threshold", type=float, default=0.5, help="Positive threshold")
    parser.add_argument("--soft_weights", nargs="+", type=float, default=None, help="Soft-voting weights")
    parser.add_argument("--hard_tie_break", default="soft",
                        choices=["soft", "highest_confidence", "label_0", "label_1"],
                        help="Tie-break method for even-number hard voting")
    parser.add_argument("--primary_idx", type=int, default=0, help="Primary model index for hierarchical")
    parser.add_argument("--secondary_indices", nargs="+", type=int, default=None,
                        help="Secondary model indices for hierarchical")
    parser.add_argument("--hierarchical_low", type=float, default=0.3, help="Primary low-confidence threshold")
    parser.add_argument("--hierarchical_high", type=float, default=0.7, help="Primary high-confidence threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.pred_files) != len(args.names):
        raise ValueError("--pred_files and --names must have the same length")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    template = load_template(args.template_csv)
    models = [
        load_prediction_file(csv_path, name)
        for csv_path, name in zip(args.pred_files, args.names)
    ]
    validate_against_template(models, template)

    names = [model.name for model in models]
    aligned = build_aligned_table(models, template)
    aligned_path = os.path.join(args.output_dir, f"ensemble_aligned_inputs_{timestamp}.csv")
    aligned.to_csv(aligned_path, index=False)

    report = {
        "timestamp": timestamp,
        "template_csv": args.template_csv,
        "num_images": int(len(template)),
        "models": [{"name": model.name, "path": model.path} for model in models],
        "aligned_inputs": aligned_path,
        "model_disagreement": disagreement_report(aligned, names),
        "strategies": {},
    }

    if "hard" in args.strategies:
        hard = hard_voting(
            aligned,
            names,
            tie_break=args.hard_tie_break,
            threshold=args.threshold,
        )
        summary = save_strategy_outputs(hard, aligned, names, args.output_dir, "hard", timestamp)
        summary.update({"tie_break": args.hard_tie_break})
        report["strategies"]["hard"] = summary

    if "soft" in args.strategies:
        soft = soft_voting(
            aligned,
            names,
            weights=args.soft_weights,
            threshold=args.threshold,
        )
        summary = save_strategy_outputs(soft, aligned, names, args.output_dir, "soft", timestamp)
        summary.update({
            "threshold": args.threshold,
            "weights": normalize_weights(args.soft_weights, len(names)).tolist(),
        })
        report["strategies"]["soft"] = summary

    if "hierarchical" in args.strategies:
        hierarchical = hierarchical_ensemble(
            aligned,
            names,
            primary_idx=args.primary_idx,
            secondary_indices=args.secondary_indices,
            low=args.hierarchical_low,
            high=args.hierarchical_high,
            threshold=args.threshold,
        )
        summary = save_strategy_outputs(
            hierarchical, aligned, names, args.output_dir, "hierarchical", timestamp
        )
        secondary_indices = args.secondary_indices
        if secondary_indices is None:
            secondary_indices = [i for i in range(len(names)) if i != args.primary_idx]
        summary.update({
            "primary_model": names[args.primary_idx],
            "secondary_models": [names[i] for i in secondary_indices],
            "low_threshold": args.hierarchical_low,
            "high_threshold": args.hierarchical_high,
            "primary_decided": int((hierarchical["source"] != "secondary_soft").sum()),
            "secondary_decided": int((hierarchical["source"] == "secondary_soft").sum()),
        })
        report["strategies"]["hierarchical"] = summary

    if "single" in args.strategies:
        report["strategies"]["single"] = {}
        for name in names:
            result = single_model(aligned, name)
            summary = save_strategy_outputs(result, aligned, names, args.output_dir, f"single_{name}", timestamp)
            report["strategies"]["single"][name] = summary

    report_path = os.path.join(args.output_dir, f"ensemble_report_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"Loaded {len(models)} models and {len(template)} template ids")
    print(f"Aligned inputs: {aligned_path}")
    print(f"Report: {report_path}")
    print("\nGenerated submissions:")
    for strategy, summary in report["strategies"].items():
        if strategy == "single":
            for model_name, model_summary in summary.items():
                print(f"  single:{model_name}: {model_summary['submission']}")
        else:
            print(f"  {strategy}: {summary['submission']}")


if __name__ == "__main__":
    main()
