"""
add_columns.py — Post-processor for zero_shot_CoT.py results CSVs
=================================================================
Reads an existing results CSV (produced by zero_shot_CoT.py) and inserts
three new columns immediately after model_reasoning:

  step1_extracted_params  – dict of {entity_name: evidence_value} from step1
  step2_computed_value    – numeric correct_result only (units stripped)
  step3_verdict           – {step1_label, step2_label, derived_label}

Usage:
    python add_columns.py --input cot_zs_llama_merged.csv
    python add_columns.py --input cot_zs_llama_merged.csv --output enriched.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Core extraction helper
# ---------------------------------------------------------------------------

def _extract_display_columns(reasoning_str: str) -> dict:
    """
    Parse model_reasoning JSON and return the three derived columns.
    Falls back to empty strings on any parse failure so the CSV row
    is never dropped.

    Input JSON shape (produced by zero_shot_CoT.py):
        {
          "entities":    [ {"name": ..., "evidence_value": ..., "correct": ...}, ... ],
          "calculation": { "correct_result": ..., "correct": ... },
          "derived_label": "true|partially true|false"
        }

    Returns:
        {
          "step1_extracted_params": '{"Chloride": 89, "Albumin": 3.9, ...}',
          "step2_computed_value":   "-22.5",
          "step3_verdict":          '{"step1_label": "false", "step2_label": "false",
                                      "derived_label": "false"}',
        }
    """
    _EMPTY = {
        "step1_extracted_params": "",
        "step2_computed_value":   "",
        "step3_verdict":          "",
    }

    if not reasoning_str or not reasoning_str.strip():
        return _EMPTY

    try:
        data = json.loads(reasoning_str)
    except (json.JSONDecodeError, TypeError):
        return _EMPTY

    entities = data.get("entities", [])
    calc     = data.get("calculation", {})

    # ── step1_extracted_params ────────────────────────────────────────────
    # Map each entity's name → evidence_value (preserve insertion order).
    step1_params: dict = {}
    for e in entities:
        name = e.get("name", "").strip()
        if name:
            step1_params[name] = e.get("evidence_value", "")

    # ── step2_computed_value ──────────────────────────────────────────────
    # Extract the leading number from correct_result, discarding units.
    # Handles values like "-22.5", "7.84 mg/dL", "5 points", "03/25/2024".
    cr_str = str(calc.get("correct_result", "")).strip()
    num_m  = re.search(r"-?\d+\.?\d*", cr_str)
    step2_value = num_m.group() if num_m else cr_str   # fall back to raw if no number

    # ── step3_verdict ─────────────────────────────────────────────────────
    # step1_label: "true" only when every entity has correct=true
    step1_all_correct = (
        bool(entities)
        and all(bool(e.get("correct", False)) for e in entities)
    )
    # step2_label: mirrors calculation.correct
    step2_correct = bool(calc.get("correct", False))

    step3_verdict = {
        "step1_label":   "true" if step1_all_correct else "false",
        "step2_label":   "true" if step2_correct     else "false",
        "derived_label": data.get("derived_label", ""),
    }

    return {
        "step1_extracted_params": json.dumps(step1_params,  ensure_ascii=False),
        "step2_computed_value":   step2_value,
        "step3_verdict":          json.dumps(step3_verdict, ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# CSV column insertion
# ---------------------------------------------------------------------------

_NEW_COLS = ["step1_extracted_params", "step2_computed_value", "step3_verdict"]

_INSERT_AFTER = "model_reasoning"


def _build_fieldnames(original: list[str]) -> list[str]:
    """
    Insert the three new columns right after model_reasoning.
    If model_reasoning is absent, append at the end.
    """
    if _INSERT_AFTER in original:
        pos = original.index(_INSERT_AFTER) + 1
        return original[:pos] + _NEW_COLS + original[pos:]
    return original + _NEW_COLS


def process(input_path: str, output_path: str) -> None:
    src = Path(input_path)
    if not src.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    dst = Path(output_path)

    total = skipped = 0

    with (
        open(src, newline="", encoding="utf-8-sig") as fin,
        open(dst, "w",  newline="", encoding="utf-8-sig") as fout,
    ):
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            print("Error: CSV has no header row.", file=sys.stderr)
            sys.exit(1)

        out_fields = _build_fieldnames(list(reader.fieldnames))
        writer     = csv.DictWriter(fout, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()

        for row in reader:
            total += 1
            reasoning = row.get("model_reasoning", "")
            derived   = _extract_display_columns(reasoning)

            if not any(derived.values()):
                skipped += 1

            row.update(derived)
            writer.writerow(row)

    print(f"Done.  {total} rows processed, {skipped} had unparseable reasoning.")
    print(f"Output written to: {dst}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add step1/step2/step3 display columns to a zero_shot_CoT results CSV."
    )
    parser.add_argument("--input",  required=True, help="Path to the input results CSV")
    parser.add_argument("--output", default="",    help="Output path (default: <input>_enriched.csv)")
    args = parser.parse_args()

    out = args.output or str(Path(args.input).with_suffix("")) + "_enriched.csv"
    process(args.input, out)


if __name__ == "__main__":
    main()
