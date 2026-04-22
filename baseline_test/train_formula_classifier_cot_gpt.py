"""
Formula Classifier — Module 3 of MedPoT
========================================
Trains a lightweight sequence classifier (BioClinicalBERT or bert-base-uncased)
on claim text → calculator ID.

The 5 QTc variants (Bazett, Framingham, Fridericia, Hodges, Rautaharju) share
identical inputs and indistinguishable claim text, so they are merged into one
class: "QTc_variant". After classification, whichever specific variant is named
in the claim's output label is used; if none is found, Bazett is the default
(most common clinical standard).

Sampling strategy:
  - Each calculator is capped at 20 samples per label (true / false / partially true),
    giving at most 60 instances per calculator.
  - The 5 QTc variants are first downsampled to 12 instances per variant (60 total),
    then merged into the single "QTc_variant" class before the per-label cap is applied.

Usage:
    # Train
    python train_formula_classifier.py \
        --data mrt_claim_cleaned.csv \
        --model_out ./formula_classifier \
        --encoder dmis-lab/biobert-base-cased-v1.2

    # Evaluate only (loads saved classifier)
    python train_formula_classifier.py \
        --data mrt_claim_cleaned.csv \
        --model_out ./formula_classifier \
        --eval_only

Dependencies:
    pip install transformers torch scikit-learn pandas
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# QTc variants that are indistinguishable from claim text alone.
# They are collapsed into a single training class.
QTC_VARIANTS = {
    "QTc Bazett Calculator",
    "QTc Framingham Calculator",
    "QTc Fridericia Calculator",
    "QTc Hodges Calculator",
    "QTc Rautaharju Calculator",
}
QTC_MERGED_LABEL = "QTc_variant"

# Keywords used to resolve the specific QTc variant post-classification.
# Order matters: more specific strings first.
QTC_RESOLUTION = [
    ("framingham", "QTc Framingham Calculator"),
    ("fridericia",  "QTc Fridericia Calculator"),
    ("hodges",      "QTc Hodges Calculator"),
    ("rautaharju",  "QTc Rautaharju Calculator"),
    ("bazett",      "QTc Bazett Calculator"),  # default fallback
]

# Sampling caps
QTC_SAMPLES_PER_VARIANT = 12   # 12 × 5 variants = 60 total for QTc_variant
SAMPLES_PER_LABEL       = 20   # cap per (calculator, label) → max 60 per calculator

# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def merge_qtc(calculator_name: str) -> str:
    """Replace any QTc variant with the merged class label."""
    if calculator_name in QTC_VARIANTS:
        return QTC_MERGED_LABEL
    return calculator_name


def resolve_qtc_variant(claim: str) -> str:
    """
    After the classifier predicts QTc_variant, inspect the claim text to find
    which specific formula is named. Falls back to Bazett if none is found.
    """
    claim_lower = claim.lower()
    for keyword, calc_name in QTC_RESOLUTION:
        if keyword in claim_lower:
            return calc_name
    return "QTc Bazett Calculator"  # clinical default

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ClaimDataset(Dataset):
    """
    Each item is (claim_text, class_index).
    Input to the classifier is the claim string only — at Module 3 runtime,
    the claim has already been decomposed so this is the natural input.
    """

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(csv_path: str):
    """
    Applies three steps before building the label vocabulary:

      1. QTc downsampling: each of the 5 QTc variants is capped at
         QTC_SAMPLES_PER_VARIANT (12) rows, giving 60 total, matching
         the count of other calculators.

      2. QTc merging: all 5 variants are relabelled as "QTc_variant".

      3. Per-label cap: for every (calculator, label) group, at most
         SAMPLES_PER_LABEL (20) rows are kept, so each calculator has
         at most 60 instances and all three labels are equally represented.

    Returns:
        texts    : list of claim strings
        labels   : list of integer class indices
        label2id : dict  calculator_name -> int
        id2label : dict  int -> calculator_name
    """
    df = pd.read_csv(csv_path)

# ---- Step 1: downsample QTc variants --------------------------------
    qtc_mask = df["Calculator Name"].isin(QTC_VARIANTS)
    qtc_df = (
        df[qtc_mask]
        .groupby("Calculator Name", group_keys=False)
        .apply(lambda x: x.sample(n=QTC_SAMPLES_PER_VARIANT, random_state=42))
    )
    non_qtc_df = df[~qtc_mask]
    df = pd.concat([non_qtc_df, qtc_df]).reset_index(drop=True)

    # ---- Step 2: merge QTc variants into one class ----------------------
    # Assign AFTER concat so the column is never dropped by groupby/apply
    df = df.copy()
    df["merged_calc"] = df["Calculator Name"].apply(merge_qtc)

    # ---- Step 3: per-label cap (20 per label per calculator) ------------
    df = (
        df.groupby(["merged_calc", "Label"], group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), SAMPLES_PER_LABEL), random_state=42))
    )
    df = df.reset_index(drop=True)

    # Re-assign after the second groupby/apply for the same reason
    df["merged_calc"] = df["Calculator Name"].apply(merge_qtc)

    texts = df["Claim"].tolist()
    calcs = df["merged_calc"].tolist()

    # Build label vocab (sorted for reproducibility)
    classes  = sorted(set(calcs))
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}
    labels   = [label2id[c] for c in calcs]

    print(f"Loaded {len(texts)} samples across {len(classes)} classes "
          f"(after QTc downsampling + per-label cap of {SAMPLES_PER_LABEL})")
    print(f"  QTc variants merged → '{QTC_MERGED_LABEL}' (resolves post-hoc)")
    print(f"\n  Class distribution:")
    for cls, cnt in sorted(Counter(calcs).items()):
        print(f"    {cls:<55} {cnt}")

    return texts, labels, label2id, id2label

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args, texts, labels, label2id, id2label):
    num_labels = len(label2id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}  |  Classes: {num_labels}")

    # Stratified split so every calculator appears in both train and val
    X_tr, X_val, y_tr, y_val = train_test_split(
        texts, labels,
        test_size=0.15,
        stratify=labels,
        random_state=42,
    )
    print(f"Train: {len(X_tr)}  |  Val: {len(X_val)}")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    tr_ds  = ClaimDataset(X_tr,  y_tr,  tokenizer, max_length=args.max_length)
    val_ds = ClaimDataset(X_val, y_val, tokenizer, max_length=args.max_length)

    tr_loader  = DataLoader(tr_ds,  batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Use AutoConfig to set num_labels correctly, avoiding the conflict
    # between the pre-trained checkpoint's binary head and our 47-class task
    config = AutoConfig.from_pretrained(
        args.encoder,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.encoder,
        config=config,
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(tr_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0
    best_preds, best_true = [], []

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        from tqdm import tqdm
        for batch in tqdm(tr_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total   += len(batch["labels"])

        train_acc = correct / total

        # --- validate ---
        model.eval()
        val_correct, val_total = 0, 0
        epoch_preds, epoch_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                val_correct += (preds == batch["labels"]).sum().item()
                val_total   += len(batch["labels"])
                epoch_preds.extend(preds.cpu().tolist())
                epoch_true.extend(batch["labels"].cpu().tolist())

        val_acc = val_correct / val_total
        print(f"Epoch {epoch}/{args.epochs}  "
              f"loss={total_loss/len(tr_loader):.4f}  "
              f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_preds   = epoch_preds
            best_true    = epoch_true
            model.save_pretrained(args.model_out)
            tokenizer.save_pretrained(args.model_out)
            print(f"  ✓ Saved best model to {args.model_out}")

    print(f"\nBest validation accuracy: {best_val_acc:.3f}")

    # Detailed report on val set from the best epoch
    print("\nClassification report (val set, best checkpoint):")
    label_names = [id2label[i] for i in sorted(id2label)]
    print(classification_report(best_true, best_preds, target_names=label_names))

    # Save label maps alongside the model
    with open(os.path.join(args.model_out, "label_map.json"), "w") as f:
        json.dump(
            {"label2id": label2id,
             "id2label": {str(k): v for k, v in id2label.items()}},
            f, indent=2,
        )

# ---------------------------------------------------------------------------
# Inference helper (used at MedPoT Module 3 runtime)
# ---------------------------------------------------------------------------

class FormulaClassifier:
    """
    Thin wrapper around the saved classifier.
    Drop this class directly into your MedPoT pipeline.

    Usage in Module 3:
        classifier = FormulaClassifier("./formula_classifier")
        calc_name  = classifier.predict(claim_text)
        # calc_name is a single calculator name string, e.g. "MDRD GFR Equation"
        # For QTc claims it resolves the specific variant from the claim text.
    """

    def __init__(self, model_dir: str):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = str(Path(model_dir).resolve())
        # transformers 5.x validates absolute paths as HF repo IDs — force offline mode
        _prev = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model     = AutoModelForSequenceClassification.from_pretrained(
                model_dir
            ).to(self.device)
        finally:
            if _prev is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = _prev
        self.model.eval()

        with open(os.path.join(model_dir, "label_map.json")) as f:
            maps = json.load(f)
        self.id2label = {int(k): v for k, v in maps["id2label"].items()}

    def predict(self, claim: str, return_probs: bool = False):
        """
        Args:
            claim        : the claim string from Module 1 decomposition
            return_probs : if True, also returns a dict of {label: prob}

        Returns:
            calculator_name (str)  — resolved calculator, e.g. "QTc Bazett Calculator"
            [optional] probs dict
        """
        enc = self.tokenizer(
            claim,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**enc).logits
        probs   = torch.softmax(logits, dim=-1)[0]
        pred_id = probs.argmax().item()
        pred_label = self.id2label[pred_id]

        # Post-hoc resolution: if the merged QTc class was predicted,
        # look at the claim text to find the specific variant
        if pred_label == QTC_MERGED_LABEL:
            pred_label = resolve_qtc_variant(claim)

        if return_probs:
            prob_dict = {
                self.id2label[i]: round(probs[i].item(), 4)
                for i in range(len(probs))
            }
            return pred_label, prob_dict

        return pred_label

# ---------------------------------------------------------------------------
# Evaluation only mode
# ---------------------------------------------------------------------------

def evaluate(args, texts, labels, label2id, id2label):
    classifier  = FormulaClassifier(args.model_out)
    preds       = [classifier.predict(t) for t in texts]
    pred_ids    = [label2id.get(p, -1) for p in preds]
    label_names = [id2label[i] for i in sorted(id2label)]
    print(classification_report(labels, pred_ids, target_names=label_names))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train/evaluate MedPoT formula classifier")
    parser.add_argument("--data",       required=True,
                        help="Path to mrt_claim_cleaned.csv")
    parser.add_argument("--model_out",  default="./formula_classifier",
                        help="Directory to save/load the classifier")
    parser.add_argument("--encoder",    default="dmis-lab/biobert-base-cased-v1.2",
                        help="HuggingFace encoder backbone. Alternatives: "
                             "emilyalsentzer/Bio_ClinicalBERT, bert-base-uncased")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--max_length", type=int,   default=256,
                        help="Max token length for claim (256 is sufficient for short claims)")
    parser.add_argument("--eval_only",  action="store_true",
                        help="Skip training, load saved model and evaluate")
    args = parser.parse_args()

    texts, labels, label2id, id2label = load_data(args.data)

    if args.eval_only:
        evaluate(args, texts, labels, label2id, id2label)
    else:
        os.makedirs(args.model_out, exist_ok=True)
        train(args, texts, labels, label2id, id2label)

if __name__ == "__main__":
    main()