"""
Step 1: Data Preparation — CSV → JSONL

Purpose:
  Converts medcalc_train_claim_full.csv into two sets of JSONL files:
  - SFT format:  full CoT trace in assistant turn (for supervised fine-tuning)
  - GRPO format: user message only, label stored separately (for RL training)

Output files:
  data/sft_train.jsonl, data/sft_val.jsonl, data/sft_test.jsonl
  data/grpo_train.jsonl, data/grpo_val.jsonl, data/grpo_test.jsonl
"""

import csv
import json
import os
import random
from collections import defaultdict

random.seed(42)

# ── System prompt ────────────────────────────────────────────────────────────
# Tell the model what the task is, what the reasoning steps are, and what the output format is
# This prompt is exactly the same in both the SFT and GRPO stages, ensuring consistency
SYSTEM_PROMPT = """\
You are a medical claim verification expert. Given a patient's clinical note (Evidence) \
and a medical numerical claim (Claim), verify the claim step by step:
1. Check each entity value stated in the claim against the patient note (Extractive Verification)
2. Verify whether the calculation result in the claim is correct given the entities (Calculation Verification)
3. Conclude with exactly one of: true / partially true / false
   - true: all entities are correct AND the calculation result is correct (within 5%)
   - partially true: all entities are correct BUT the calculation result is wrong
   - false: one or more entity values are wrong (and the calculation result is also wrong)

Always reason step by step inside <think>...</think> before giving your final answer \
inside <answer>...</answer>.\
"""


def build_user_message(evidence: str, claim: str) -> str:
    """User message: patient note + claim to verify."""
    return f"Evidence:\n{evidence}\n\nClaim:\n{claim}"


def build_sft_assistant(explanation: str, label: str) -> str:
    """
    Assistant message for SFT: wrap existing Explanation as CoT.

    The Explanation from claim_generator.py is already structured:
      Extractive Claim Verification: ...
      Implicit Calculation Claim Verification: ...
      Ground Truth Explanation: ...

    We wrap it in <think> tags so the model learns this reasoning format,
    then append the final <answer> tag.
    """
    return f"<think>\n{explanation.strip()}\n</think>\n<answer>{label}</answer>"


def make_sft_record(evidence: str, claim: str, explanation: str, label: str) -> dict:
    """One training example for CoT SFT — includes full reasoning trace."""
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": build_user_message(evidence, claim)},
            {"role": "assistant", "content": build_sft_assistant(explanation, label)},
        ],
        "label": label,
    }


def make_grpo_record(evidence: str, claim: str, label: str) -> dict:
    """
    One training example for GRPO — NO CoT in assistant turn.

    During GRPO, the model must generate its own reasoning from scratch.
    We only provide the prompt; the reward function evaluates the output.
    The 'label' field is used by the reward function (not shown to the model).
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_message(evidence, claim)},
        ],
        "label": label,
    }


def stratified_split(rows_by_label: dict, train_ratio=0.8, val_ratio=0.1):
    """
    Stratified split by label so each split has the same class distribution.

    Why stratified?
      Our dataset is already perfectly balanced (33.3% each label).
      Stratified splitting ensures train/val/test are also balanced,
      preventing evaluation bias.
    """
    train, val, test = [], [], []
    for label, rows in rows_by_label.items():
        random.shuffle(rows)
        n = len(rows)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        train.extend(rows[:n_train])
        val.extend(rows[n_train:n_train + n_val])
        test.extend(rows[n_train + n_val:])
    # Shuffle within each split to mix labels
    for split in (train, val, test):
        random.shuffle(split)
    return train, val, test


def write_jsonl(records: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):,} records → {path}")


def main(
    input_csv: str = "medcalc_train_claim_full.csv",
    output_dir: str = "data",
):
    print(f"Loading {input_csv} ...")
    rows_by_label = defaultdict(list)

    with open(input_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_label[row["Label"]].append(row)

    total = sum(len(v) for v in rows_by_label.values())
    print(f"Total rows: {total:,}")
    for label, rows in rows_by_label.items():
        print(f"  {label}: {len(rows):,}")

    # ── Stratified split ────────────────────────────────────────────────────
    print("\nSplitting (stratified 80/10/10) ...")
    train_rows, val_rows, test_rows = stratified_split(rows_by_label)
    print(f"  train={len(train_rows):,}  val={len(val_rows):,}  test={len(test_rows):,}")

    # ── Build SFT records (include Explanation as CoT) ──────────────────────
    print("\nBuilding SFT records ...")
    sft_train = [make_sft_record(r["Evidence"], r["Claim"], r["Explanation"], r["Label"]) for r in train_rows]
    sft_val   = [make_sft_record(r["Evidence"], r["Claim"], r["Explanation"], r["Label"]) for r in val_rows]
    sft_test  = [make_sft_record(r["Evidence"], r["Claim"], r["Explanation"], r["Label"]) for r in test_rows]

    write_jsonl(sft_train, os.path.join(output_dir, "sft_train.jsonl"))
    write_jsonl(sft_val,   os.path.join(output_dir, "sft_val.jsonl"))
    write_jsonl(sft_test,  os.path.join(output_dir, "sft_test.jsonl"))

    # ── Build GRPO records (no CoT — model must reason itself) ──────────────
    print("\nBuilding GRPO records ...")
    grpo_train = [make_grpo_record(r["Evidence"], r["Claim"], r["Label"]) for r in train_rows]
    grpo_val   = [make_grpo_record(r["Evidence"], r["Claim"], r["Label"]) for r in val_rows]
    grpo_test  = [make_grpo_record(r["Evidence"], r["Claim"], r["Label"]) for r in test_rows]

    write_jsonl(grpo_train, os.path.join(output_dir, "grpo_train.jsonl"))
    write_jsonl(grpo_val,   os.path.join(output_dir, "grpo_val.jsonl"))
    write_jsonl(grpo_test,  os.path.join(output_dir, "grpo_test.jsonl"))

    # ── Sanity check: print one example of each type ────────────────────────
    print("\n── SFT sample (true label) ──")
    sample = next(r for r in sft_train if r["label"] == "true")
    print("USER :", sample["messages"][1]["content"][:200], "...")
    print("ASST :", sample["messages"][2]["content"][:300], "...")

    print("\n── GRPO sample (false label) ──")
    sample = next(r for r in grpo_train if r["label"] == "false")
    print("USER  :", sample["messages"][1]["content"][:200], "...")
    print("LABEL :", sample["label"])

    print("\nDone. Data saved to ./data/")


if __name__ == "__main__":
    main()
