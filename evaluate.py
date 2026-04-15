"""
Evaluation script for NLI Medical Numerical Verification results.
Computes macro and weighted Precision, Recall, F1 for classification outputs,
plus Task Fulfillment (TF), Calculator Selection (CS), and Quantitative Precision (QP)
as defined in the paper (Appendix A.2).

Usage:
    python evaluate.py
"""

import csv
from collections import Counter


def load_rows(filepath):
    """Load all rows (as dicts) from a CSV file."""
    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_labels(filepath):
    """Load true_label and predicted_label columns from a CSV file."""
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    true_labels = [r["true_label"].strip().lower() for r in rows]
    pred_labels = [r["predicted_label"].strip().lower() for r in rows]
    return true_labels, pred_labels


def compute_metrics(true_labels, pred_labels):
    """
    Compute per-class and aggregate P, R, F1.

    Unknown predictions are kept as wrong predictions (penalized) rather than
    skipped. They never match any true label, so they contribute to FP/FN.

    Parameters
    ----------
    true_labels : list[str]
    pred_labels : list[str]

    Returns
    -------
    dict with per-class metrics, macro averages, weighted averages, counts.
    """
    unknown_count = sum(1 for p in pred_labels if p == "unknown")

    # Classes derived only from true labels (unknown is not a valid class)
    classes = sorted(set(true_labels))
    n = len(true_labels)

    per_class = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)
        support = sum(1 for t in true_labels if t == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    # Macro: unweighted mean over classes
    macro_p = sum(v["precision"] for v in per_class.values()) / len(classes)
    macro_r = sum(v["recall"]    for v in per_class.values()) / len(classes)
    macro_f1 = sum(v["f1"]       for v in per_class.values()) / len(classes)

    # Weighted: weighted by support (true count per class)
    total_support = sum(v["support"] for v in per_class.values())
    weighted_p  = sum(v["precision"] * v["support"] for v in per_class.values()) / total_support
    weighted_r  = sum(v["recall"]    * v["support"] for v in per_class.values()) / total_support
    weighted_f1 = sum(v["f1"]        * v["support"] for v in per_class.values()) / total_support

    accuracy = sum(1 for t, p in zip(true_labels, pred_labels) if t == p) / n if n else 0.0

    return {
        "per_class": per_class,
        "macro":    {"precision": macro_p,    "recall": macro_r,    "f1": macro_f1},
        "weighted": {"precision": weighted_p, "recall": weighted_r, "f1": weighted_f1},
        "accuracy": accuracy,
        "n_total": n,
        "n_unknown": unknown_count,
    }


def print_report(name, results):
    pc = results["per_class"]
    classes = sorted(pc.keys())

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Total samples : {results['n_total']}")
    print(f"  Unknown preds : {results['n_unknown']} (treated as wrong)")
    print(f"  Accuracy      : {results['accuracy']:.4f}")
    print()

    # Per-class table
    col_w = max(len(c) for c in classes) + 2
    header = f"  {'Class':<{col_w}}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  {'Support':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cls in classes:
        v = pc[cls]
        print(f"  {cls:<{col_w}}  {v['precision']:>10.4f}  {v['recall']:>10.4f}  {v['f1']:>10.4f}  {v['support']:>8}")

    print()
    print(f"  {'Macro avg':<{col_w}}  {results['macro']['precision']:>10.4f}  {results['macro']['recall']:>10.4f}  {results['macro']['f1']:>10.4f}")
    print(f"  {'Weighted avg':<{col_w}}  {results['weighted']['precision']:>10.4f}  {results['weighted']['recall']:>10.4f}  {results['weighted']['f1']:>10.4f}")
    print()


# ---------------------------------------------------------------------------
# Paper metrics: TF, CS, QP  (Appendix A.2)
# ---------------------------------------------------------------------------

def compute_task_fulfillment(rows, pred_col="predicted_label"):
    """
    Task Fulfillment (TF) = N_success / N_total * 100%

    N_success: rows with a valid (non-unknown) final predicted label.
    'unknown' means the model failed to produce a valid output for that task.
    """
    n_total = len(rows)
    n_success = sum(1 for r in rows if r[pred_col].strip().lower() != "unknown")
    return {
        "TF": n_success / n_total * 100 if n_total else 0.0,
        "n_success": n_success,
        "n_total": n_total,
    }


def compute_calculator_selection(rows, pred_calc_col, gt_calc_col):
    """
    Calculator Selection (CS) = (1/N_total) * sum_i(N_correct_calc_i / N_calc_i) * 100%

    Each row is one task requiring exactly one calculator (N_calc_i = 1), so this
    simplifies to: CS = mean(predicted_calc_id == gt_calc_id) * 100%
    """
    n_total = len(rows)
    if n_total == 0:
        return {"CS": 0.0, "n_correct": 0, "n_total": 0}
    n_correct = sum(
        1 for r in rows
        if r[pred_calc_col].strip() == r[gt_calc_col].strip()
    )
    return {"CS": n_correct / n_total * 100, "n_correct": n_correct, "n_total": n_total}


def compute_quantitative_precision(rows, pred_val_col, lower_col, upper_col):
    """
    Quantitative Precision (QP) = (1/N_total) * sum_i [1/N_calc_i * sum_j 1(|y-yhat|<=eps)] * 100%

    Each row is one task with one calculator (N_calc_i = 1).
    A value is correct if: lower_limit <= computed_value <= upper_limit.
    Rows with missing or unparseable computed values count as incorrect.
    """
    n_total = len(rows)
    if n_total == 0:
        return {"QP": 0.0, "n_within": 0, "n_total": 0, "n_missing": 0}
    n_within = 0
    n_missing = 0
    for r in rows:
        try:
            val = float(r[pred_val_col].strip())
            lo  = float(r[lower_col].strip())
            hi  = float(r[upper_col].strip())
            if lo <= val <= hi:
                n_within += 1
        except (ValueError, TypeError, AttributeError):
            n_missing += 1
    return {
        "QP": n_within / n_total * 100,
        "n_within": n_within,
        "n_total": n_total,
        "n_missing": n_missing,
    }


def join_gt_columns_by_position(rows, gt_rows, columns):
    """
    Merge columns from gt_rows into rows by position (both files share the same row order).
    Joined columns are prefixed with '__gt_' to avoid name collisions.
    """
    result = []
    for r, gt_r in zip(rows, gt_rows):
        merged = dict(r)
        for col in columns:
            merged[f"__gt_{col}"] = gt_r.get(col, "")
        result.append(merged)
    return result


def print_paper_metrics(name, tf=None, cs=None, qp=None):
    print(f"  --- Paper Metrics ({name}) ---")
    if tf is not None:
        print(f"  TF (Task Fulfillment)    : {tf['TF']:6.2f}%  "
              f"({tf['n_success']}/{tf['n_total']} tasks with valid output)")
    else:
        print("  TF (Task Fulfillment)    : N/A")

    if cs is not None:
        print(f"  CS (Calculator Selection): {cs['CS']:6.2f}%  "
              f"({cs['n_correct']}/{cs['n_total']} correct)")
    else:
        print("  CS (Calculator Selection): N/A (no calculator selection in this pipeline)")

    if qp is not None:
        print(f"  QP (Quantitative Prec.)  : {qp['QP']:6.2f}%  "
              f"({qp['n_within']}/{qp['n_total']} within tolerance, "
              f"{qp['n_missing']} missing/unparseable)")
    else:
        print("  QP (Quantitative Prec.)  : N/A (no computed numeric values)")
    print()


# ---------------------------------------------------------------------------
# File configuration
# ---------------------------------------------------------------------------

# Ground-truth dataset: provides Calculator ID, Lower Limit, Upper Limit per row.
# Row ordering matches cot_clf_mrt_full.csv (both have 2814 rows).
GT_FILE = "mrt_claim_cleaned.csv"

FILES = {
    "ZS-CoT (zs_cot_mrt_770)":   "results/zs_cot_mrt_770.csv",
    "CoT-CLF (cot_clf_mrt_full)": "results/cot_clf_mrt_full.csv",
    "PoT-CLF (pot_clf_mrt_full)": "results/pot_clf_mrt_full.csv",
}

if __name__ == "__main__":
    gt_rows = load_rows(GT_FILE)

    for name, path in FILES.items():
        true_labels, pred_labels = load_labels(path)
        results = compute_metrics(true_labels, pred_labels)
        print_report(name, results)

        rows = load_rows(path)
        tf = compute_task_fulfillment(rows)

        if name == "ZS-CoT (zs_cot_mrt_770)":
            # No calculator selection step; no computed numeric values.
            print_paper_metrics(name, tf=tf, cs=None, qp=None)

        elif name in ("CoT-CLF (cot_clf_mrt_full)", "PoT-CLF (pot_clf_mrt_full)"):
            # Join GT columns (Calculator ID, Lower Limit, Upper Limit) by row index.
            # cot_clf row "index" matches GT "Row Number".
            joined = join_gt_columns_by_position(
                rows, gt_rows,
                columns=["Calculator ID", "Lower Limit", "Upper Limit"],
            )
            cs = compute_calculator_selection(
                joined, "classifier_calculator_id", "__gt_Calculator ID"
            )
            qp = compute_quantitative_precision(
                joined, "step2_computed_value", "__gt_Lower Limit", "__gt_Upper Limit"
            )
            print_paper_metrics(name, tf=tf, cs=cs, qp=qp)
