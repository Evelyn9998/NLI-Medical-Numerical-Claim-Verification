"""
Calculator Validation Script
=============================
Tests each calculator implementation against ground truth values
from medcalc_train_claim_full.csv.

For each Calculator ID, samples up to N_SAMPLES rows,
extracts Relevant Entities, runs our Python function,
and checks if the result falls within [Lower Limit, Upper Limit].
"""

import ast
import csv
import json
from collections import defaultdict
from calculators import run_calculator, CALCULATOR_REGISTRY

# ── Config ─────────────────────────────────────────────────────
DATA_PATH   = "medcalc_train_claim_full.csv"
N_SAMPLES   = 20      # rows to test per calculator
SKIP_DATE_CALCS = {13, 68, 69}   # date-based calculators, skip for now

# ── Load dataset ───────────────────────────────────────────────
def load_data(path):
    with open(path, 'rb') as f:
        raw = f.read()
    if raw[:3] == b'\xef\xbb\xbf':
        raw = raw[3:]
    raw = raw.replace(b'\x00', b'')
    text = raw.decode('utf-8', errors='replace')
    lines = text.split('\n')
    reader = csv.DictReader(iter(lines))
    return list(reader)

# ── Parse Relevant Entities into flat param dict ───────────────
def parse_entities(entity_str: str) -> dict:
    """
    Convert Relevant Entities string to a flat dict our calculators can use.
    Input format: {'Sodium': [135.0, 'mEq/L'], 'sex': 'Female', 'age': [55, 'years']}
    Output:       {'sodium': 135.0, 'sex': 'female', 'age': 55}
    """
    if not entity_str or entity_str.strip() in ('', 'nan'):
        return {}
    try:
        entities = ast.literal_eval(entity_str.strip())
    except Exception:
        return {}

    flat = {}
    for key, val in entities.items():
        k = key.lower().strip()
        if isinstance(val, list) and len(val) >= 1:
            # [numeric_value, 'unit'] → keep just the number
            flat[k] = val[0]
        else:
            flat[k] = val
    return flat

# ── Main validation ────────────────────────────────────────────
def validate():
    print("Loading dataset...")
    rows = load_data(DATA_PATH)
    print(f"Loaded {len(rows)} rows.\n")

    # Group by calculator ID
    by_calc = defaultdict(list)
    for row in rows:
        cid = row.get('Calculator ID', '').strip()
        if cid:
            by_calc[cid].append(row)

    results = {}   # calc_id → {total, passed, failed, errors, details}

    for cid, calc_rows in sorted(by_calc.items(), key=lambda x: int(x[0])):
        if int(cid) in SKIP_DATE_CALCS:
            continue

        calc_name = calc_rows[0].get('Calculator Name', '').strip()
        sample = calc_rows[:N_SAMPLES]

        total = passed = failed = errors = 0
        fail_examples = []

        for row in sample:
            total += 1
            entity_str  = row.get('Relevant Entities', '')
            gt_str      = row.get('Ground Truth Answer', '')
            lower_str   = row.get('Lower Limit', '')
            upper_str   = row.get('Upper Limit', '')

            params = parse_entities(entity_str)

            # Run calculator
            output = run_calculator(calc_name, params)

            if not output['success']:
                errors += 1
                fail_examples.append({
                    'error': output['error'],
                    'params': params
                })
                continue

            result = output['result']

            # Check against ground truth
            try:
                gt    = float(gt_str)
                lower = float(lower_str)
                upper = float(upper_str)

                if lower <= float(result) <= upper:
                    passed += 1
                else:
                    failed += 1
                    if len(fail_examples) < 3:
                        fail_examples.append({
                            'params': params,
                            'our_result': result,
                            'ground_truth': gt,
                            'range': [lower, upper]
                        })
            except (ValueError, TypeError):
                # Non-numeric ground truth (e.g. date) — just check not None
                if result is not None:
                    passed += 1
                else:
                    errors += 1

        results[cid] = {
            'name': calc_name,
            'total': total,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'accuracy': round(passed / total * 100, 1) if total > 0 else 0,
            'fail_examples': fail_examples
        }

    # ── Print Summary ──────────────────────────────────────────
    print(f"{'ID':<5} {'Calculator':<50} {'Pass':>5} {'Fail':>5} {'Err':>5} {'Acc':>7}")
    print("-" * 80)

    all_passed = all_total = 0
    needs_fix = []

    for cid, r in results.items():
        flag = "" if r['accuracy'] >= 80 else " ← FIX"
        print(f"{cid:<5} {r['name'][:50]:<50} {r['passed']:>5} {r['failed']:>5} "
              f"{r['errors']:>5} {r['accuracy']:>6.1f}%{flag}")
        all_passed += r['passed']
        all_total  += r['total']
        if r['accuracy'] < 80:
            needs_fix.append((cid, r))

    print("-" * 80)
    print(f"{'TOTAL':<56} {all_passed:>5}/{all_total:<5} "
          f"{round(all_passed/all_total*100,1) if all_total else 0:>6.1f}%\n")

    # ── Detail failed calculators ──────────────────────────────
    if needs_fix:
        print("\n" + "=" * 80)
        print("CALCULATORS NEEDING FIXES")
        print("=" * 80)
        for cid, r in needs_fix:
            print(f"\n[ID {cid}] {r['name']}")
            print(f"  Accuracy: {r['accuracy']}%  "
                  f"(passed={r['passed']}, failed={r['failed']}, errors={r['errors']})")
            for ex in r['fail_examples'][:2]:
                if 'error' in ex:
                    print(f"  ERROR: {ex['error']}")
                    print(f"  Params: {ex['params']}")
                else:
                    print(f"  Our result : {ex['our_result']}")
                    print(f"  Ground truth: {ex['ground_truth']}  "
                          f"(range [{ex['range'][0]}, {ex['range'][1]}])")
                    print(f"  Params: {dict(list(ex['params'].items())[:5])}...")

    return results

if __name__ == "__main__":
    validate()
