"""
validate_calculators.py
-----------------------
Validates every calculator formula in calculators.py against the dataset's
ground-truth answers.

For each row:
  1. Parse Relevant Entities -> params dict  (uses ground-truth values, bypassing LLM)
  2. Call run_calculator(id, params)
  3. Check: lower_limit <= result <= upper_limit  -> PASS / FAIL

Output:
  - Per-calculator summary (pass rate, sample count)
  - Overall accuracy
  - Optionally: detailed failure CSV

Usage:
    python validate_calculators.py
    python validate_calculators.py --data ../medcalc_train_claim_full.csv
    python validate_calculators.py --calc-id 2          # single calculator
    python validate_calculators.py --failures failures.csv
"""

import argparse
import ast
import csv
import sys
import os
from collections import defaultdict

# ── import the calculator registry ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from calculators import run_calculator


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_entities(raw: str) -> dict:
    """
    Relevant Entities field has two formats:
      - Numeric: {"Sodium": [139.0, "mEq/L"]}   → extract v[0]
      - Boolean: {"Stroke": True}                 → convert to 1/0
      - String:  {"sex": "Male"}                  → keep as-is
    """
    try:
        d = ast.literal_eval(raw.strip())
    except Exception:
        return {}
    result = {}
    for k, v in d.items():
        key = k.lower().strip()
        if isinstance(v, (list, tuple)) and len(v) >= 1:
            result[key] = v[0]
        elif isinstance(v, bool):
            result[key] = int(v)   # True → 1, False → 0
        elif isinstance(v, (int, float)):
            result[key] = v
        elif isinstance(v, str):
            result[key] = v
    return result


def load_dataset(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                calc_id   = int(row["Calculator ID"])
                gt        = float(row["Ground Truth Answer"])
                lo        = float(row["Lower Limit"])
                hi        = float(row["Upper Limit"])
                entities  = parse_entities(row["Relevant Entities"])
                calc_name = row.get("Calculator Name", "")
            except (ValueError, KeyError):
                continue
            if not entities:
                continue
            rows.append({
                "row_num":   row.get("Row Number", ""),
                "calc_id":   calc_id,
                "calc_name": calc_name,
                "params":    entities,
                "gt":        gt,
                "lo":        lo,
                "hi":        hi,
                "evidence":  row.get("Evidence", "")[:120],
            })
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate calculator formulas against ground truth")
    parser.add_argument("--data", default="../medcalc_train_claim_full.csv")
    parser.add_argument("--calc-id", type=int, default=None, help="Only validate this calculator ID")
    parser.add_argument("--failures", default="", help="Write failures to this CSV path")
    args = parser.parse_args()

    rows = load_dataset(args.data)
    if args.calc_id:
        rows = [r for r in rows if r["calc_id"] == args.calc_id]
    print(f"Loaded {len(rows)} rows" + (f" for calc_id={args.calc_id}" if args.calc_id else ""))

    # ── evaluate ──────────────────────────────────────────────────────────────
    per_calc  = defaultdict(lambda: {"pass": 0, "fail": 0, "error": 0, "name": ""})
    failures  = []
    total_pass = total_fail = total_error = 0

    for r in rows:
        cid  = r["calc_id"]
        per_calc[cid]["name"] = r["calc_name"]

        result = run_calculator(cid, r["params"])
        val    = result.get("value")

        if val is None:
            per_calc[cid]["error"] += 1
            total_error += 1
            failures.append({**r, "computed": "ERROR", "note": result.get("note", "")})
            continue

        passed = r["lo"] <= val <= r["hi"]
        if passed:
            per_calc[cid]["pass"] += 1
            total_pass += 1
        else:
            per_calc[cid]["fail"] += 1
            total_fail += 1
            failures.append({
                **r,
                "computed": round(val, 6),
                "unit":     result.get("unit", ""),
                "note":     result.get("note", ""),
            })

    # ── print summary ─────────────────────────────────────────────────────────
    total = total_pass + total_fail + total_error
    print(f"\n{'='*65}")
    print(f"  OVERALL  pass={total_pass}  fail={total_fail}  error={total_error}  "
          f"accuracy={total_pass/total*100:.1f}%  (n={total})")
    print(f"{'='*65}")

    print(f"\n{'ID':<6} {'Pass':>5} {'Fail':>5} {'Err':>5} {'Acc':>7}  Calculator")
    print("-" * 65)
    for cid in sorted(per_calc):
        s = per_calc[cid]
        n = s["pass"] + s["fail"] + s["error"]
        acc = s["pass"] / n * 100 if n else 0
        marker = "  <-- NEEDS FIX" if acc < 60 and n >= 3 else ""
        print(f"{cid:<6} {s['pass']:>5} {s['fail']:>5} {s['error']:>5} {acc:>6.0f}%  {s['name']}{marker}")

    # ── write failures CSV ────────────────────────────────────────────────────
    if args.failures and failures:
        fields = ["row_num", "calc_id", "calc_name", "gt", "lo", "hi",
                  "computed", "unit", "note", "params", "evidence"]
        with open(args.failures, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for fl in failures:
                fl["params"] = str(fl["params"])
                w.writerow(fl)
        print(f"\nFailures written to: {args.failures}")


if __name__ == "__main__":
    main()
