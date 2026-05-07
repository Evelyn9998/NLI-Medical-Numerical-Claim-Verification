"""
LLM Fact-Check Evaluator — Zero-shot Program of Thought (PoT)
Uses GPT-4o-mini (or any OpenAI chat model) via the OpenAI API.

Zero-shot PoT: Evidence + Claim -> LLM emits Python code -> host executes it
               -> structured JSON output -> Logical Reasoner -> Label.

The chain-of-thought structure (STEP 1 / STEP 2 / STEP 3) is identical to the
CoT variant, but all reasoning is expressed as executable Python rather than
natural-language JSON prose.  The host runs the generated code in a sandboxed
namespace, captures its stdout, then applies the same post-processing pipeline:

  _remove_implicit_from_step1  →  strip the calculator output from step1
  _override_step2_by_criteria  →  enforce MedCalc-Bench accuracy tolerances
  _logical_reasoner            →  derive the final label from step1 + step2

Benchmark accuracy criteria (MedCalc-Bench §3.1, Table 2):
  - Rule-based calculators  (integer scores)  : exact match
  - Equation-based calculators (decimal output): within 5 %
  - Date-based calculators  (calendar date)   : exact match

The generated program self-classifies the calculator type and applies the
correct criterion in Python.  The host's _override_step2_by_criteria provides
an additional safety net that re-checks the criterion from the Python output.

Output CSV columns:
  index | evidence | claim | true_label | predicted_label | match |
  model_reasoning | dataset_explanation

Install dependencies:
    pip install openai

Set your API key before running:
    export OPENAI_API_KEY="sk-..."

Usage:
    python zero_shot_PoT.py --data "mrt_claim_cleaned.csv"
    python zero_shot_PoT.py --data "train_300.csv" --model "gpt-4o-mini" --samples 20 --random
    python zero_shot_PoT.py --data "train_300.csv" --model "gpt-4o"      --samples 0
    python zero_shot_PoT.py --data "train_300.csv" --samples 10 --seed 42
"""

import argparse
import csv
import io
import json
import os
import re
import random
import sys
import time
import traceback
from datetime import datetime


# ---------------------------------------------------------------------------
# FormulaRAG — lightweight retrieval over web_formula.txt
# ---------------------------------------------------------------------------

class FormulaRAG:
    """
    Minimal RAG that retrieves the *k* most similar formula blocks from a
    plain-text knowledge base delimited by <<FORMULA START>> / <<FORMULA END>>.

    Indexing strategy (in preference order):
      1. scikit-learn TF-IDF + cosine similarity  (no API key needed)
      2. Pure-Python keyword overlap fallback      (zero extra deps)

    Parameters
    ----------
    doc_path : str
        Path to the formula knowledge base (e.g. ``data/web_formula.txt``).
    ngram_range : tuple
        n-gram range passed to TfidfVectorizer (default: unigrams + bigrams).
    """

    START_MARKER = "<<FORMULA START>>"
    END_MARKER   = "<<FORMULA END>>"

    def __init__(
        self,
        doc_path: str = "data/web_formula.txt",
        ngram_range: tuple = (1, 2),
    ) -> None:
        if not os.path.exists(doc_path):
            raise FileNotFoundError(
                f"Formula knowledge base not found: {doc_path}\n"
                "Pass --formula-db <path> or omit the flag to skip RAG."
            )
        self.formulas: list[str] = self._load_blocks(doc_path)
        if not self.formulas:
            raise ValueError(
                "No formula blocks found between "
                "<<FORMULA START>> and <<FORMULA END>> markers."
            )
        self._build_index(ngram_range)
        print(f"[FormulaRAG] Loaded {len(self.formulas)} formula blocks from {doc_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_blocks(self, doc_path: str) -> list[str]:
        with open(doc_path, "r", encoding="utf-8") as fh:
            text = fh.read()
        blocks, pos = [], 0
        while True:
            start = text.find(self.START_MARKER, pos)
            if start == -1:
                break
            start += len(self.START_MARKER)
            end = text.find(self.END_MARKER, start)
            if end == -1:
                break
            content = text[start:end].strip()
            if content:
                blocks.append(content)
            pos = end + len(self.END_MARKER)
        return blocks

    def _build_index(self, ngram_range: tuple) -> None:
        """Try sklearn TF-IDF; fall back to keyword overlap."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                sublinear_tf=True,
            )
            self._matrix = self._vectorizer.fit_transform(self.formulas)
            self._use_tfidf = True
            print("[FormulaRAG] Index: TF-IDF (sklearn)")
        except ImportError:
            self._use_tfidf = False
            print("[FormulaRAG] sklearn not found — using keyword overlap fallback")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 1) -> list[tuple[str, float]]:
        """
        Unweighted retrieval — scores a single query string against all blocks.
        Kept for backward compatibility and direct use.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        k = min(k, len(self.formulas))
        if self._use_tfidf:
            return self._retrieve_tfidf(query, k)
        return self._retrieve_keyword(query, k)

    def retrieve_weighted(
        self,
        last_seg: str,
        earlier_segs: str = "",
        k: int = 1,
        last_weight: float = 1.0,
        earlier_weight: float = 0.3,
    ) -> list[tuple[str, float]]:
        """
        Weighted retrieval over two query parts.

        The last claim segment (implicit calculation) drives retrieval with
        full weight; earlier segments (input parameters) contribute with a
        smaller weight so they can nudge ambiguous cases without overriding
        the primary signal.

        Final score for each formula block:
            score = last_weight  * sim(last_seg,    block)
                  + earlier_weight * sim(earlier_segs, block)

        Parameters
        ----------
        last_seg      : last comma-separated segment of the claim
                        (names the target calculator directly)
        earlier_segs  : all other segments joined into one string
                        (input parameters — softer signal)
        last_weight   : weight for last_seg scores   (default 1.0)
        earlier_weight: weight for earlier_segs scores (default 0.3)
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        k = min(k, len(self.formulas))
        if self._use_tfidf:
            return self._tfidf_weighted(last_seg, earlier_segs, k,
                                        last_weight, earlier_weight)
        return self._keyword_weighted(last_seg, earlier_segs, k,
                                      last_weight, earlier_weight)

    # ------------------------------------------------------------------
    # Internal retrieval implementations
    # ------------------------------------------------------------------

    def _retrieve_tfidf(self, query: str, k: int) -> list[tuple[str, float]]:
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec  = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_idx = scores.argsort()[::-1][:k]
        return [(self.formulas[i], float(scores[i])) for i in top_idx]

    def _tfidf_weighted(
        self,
        last_seg: str,
        earlier_segs: str,
        k: int,
        last_weight: float,
        earlier_weight: float,
    ) -> list[tuple[str, float]]:
        from sklearn.metrics.pairwise import cosine_similarity
        scores = last_weight * cosine_similarity(
            self._vectorizer.transform([last_seg]), self._matrix
        ).flatten()
        if earlier_segs.strip():
            scores = scores + earlier_weight * cosine_similarity(
                self._vectorizer.transform([earlier_segs]), self._matrix
            ).flatten()
        top_idx = scores.argsort()[::-1][:k]
        return [(self.formulas[i], float(scores[i])) for i in top_idx]

    def _retrieve_keyword(self, query: str, k: int) -> list[tuple[str, float]]:
        """Jaccard-style word-overlap (no extra deps)."""
        q_words = set(re.sub(r"[^\w\s]", " ", query.lower()).split())
        scored  = [self._jaccard(q_words, block) for block in self.formulas]
        top_idx = sorted(range(len(scored)), key=lambda i: scored[i], reverse=True)[:k]
        return [(self.formulas[i], scored[i]) for i in top_idx]

    def _keyword_weighted(
        self,
        last_seg: str,
        earlier_segs: str,
        k: int,
        last_weight: float,
        earlier_weight: float,
    ) -> list[tuple[str, float]]:
        last_words    = set(re.sub(r"[^\w\s]", " ", last_seg.lower()).split())
        earlier_words = set(re.sub(r"[^\w\s]", " ", earlier_segs.lower()).split()) \
                        if earlier_segs.strip() else set()
        scored = []
        for block in self.formulas:
            s = last_weight * self._jaccard(last_words, block)
            if earlier_words:
                s += earlier_weight * self._jaccard(earlier_words, block)
            scored.append(s)
        top_idx = sorted(range(len(scored)), key=lambda i: scored[i], reverse=True)[:k]
        return [(self.formulas[i], scored[i]) for i in top_idx]

    def _jaccard(self, q_words: set, block: str) -> float:
        b_words = set(re.sub(r"[^\w\s]", " ", block.lower()).split())
        union   = q_words | b_words
        return len(q_words & b_words) / len(union) if union else 0.0

    def __len__(self) -> int:
        return len(self.formulas)


# ---------------------------------------------------------------------------
# System prompt — instructs the LLM to emit executable Python (PoT)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a medical fact-checking assistant.
Given clinical evidence and a clinical claim, write a self-contained Python
program that determines whether the claim is TRUE, PARTIALLY TRUE, or FALSE.

The program must work through STEP 1 → STEP 2 → STEP 3, then print ONE JSON
object to stdout and nothing else.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO SPLIT THE CLAIM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Split the claim on every comma ( , ).  Each segment is one entity.
  • Every segment EXCEPT the last  →  INPUT parameters  →  handled in STEP 1.
  • The LAST segment               →  final computed result  →  STEP 2 only.

EXAMPLE:
  Claim: "...the Calcium is 7.2 mg/dL, the Albumin is 3.2 g/dL,
           the Corrected Calcium Concentration is 7.84 mg/dL."
  Segments:  Calcium | Albumin | Corrected Calcium Concentration  ← last = STEP 2
  STEP 1 entities:   Calcium (7.2 mg/dL), Albumin (3.2 g/dL)
  STEP 2 calculation: Corrected Calcium Concentration = 7.84 mg/dL

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — Verify each INPUT entity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For every INPUT segment (all except the last), look it up in the evidence and
set correct = True / False.

Special cases:
  • Positively-framed condition (e.g. "Liver disease criteria for HAS-BLED"):
      must be present in the evidence → False if not found.
  • Negatively-framed condition (e.g. "Cough Absent"):
      describes absence → True if the condition is not mentioned in the evidence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — Calculate using evidence values
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
a) Name the formula and write its expression as a Python comment.
   If a "Formula Reference" block appears in the user message, use it as
   your authoritative formula — do NOT hallucinate coefficients or structure.
b) Assign each input variable FROM THE EVIDENCE (the claim may be wrong).
   ══ RULE-BASED CALCULATORS — CRITICAL ══
   Use the evidence_* variables you already set in STEP 1 to decide each
   criterion's contribution.  NEVER use claim_* variables here.
   Criteria come in three forms — use the correct Python pattern for each:

   • Binary (yes/no present/absent):
       score += 1 if evidence_stroke == "yes" else 0

   • Threshold / range (ordered numeric bands) — use if/elif/else blocks,
     NOT nested ternary expressions, to avoid the common 'else 1' bug:
       if evidence_age >= 75:
           score += 2
       elif evidence_age >= 65:
           score += 1
       # else: 0 — no score += line needed; omitting it is correct
       ✗ WRONG:  score += 2 if age>=75 else (1 if age>=65 else 0)  ← error-prone
       ✗ WRONG:  score += 1 if age>=65 else 1                      ← always adds 1!

   • Categorical (named groups with different weights):
       score += 1 if evidence_sex.lower() == "female" else 0

   The claim_* variables show what the claim asserts — they may be wrong and
   must NEVER drive the score calculation.
c) Compute the result in Python and store it in `correct_result`.
d) Store the value stated in the claim's last segment in `claimed_result`.
e) Pick calculator_type: "rule-based" | "equation-based" | "date-based"
f) Apply the right accuracy criterion IN PYTHON:
     rule-based  / date-based  → exact match  (int or string equality)
     equation-based            → within 5 %   abs(correct - claimed)/abs(correct) ≤ 0.05

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — Derive the label IN PYTHON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  "true"          : ALL step1 entities correct=True  AND  step2 calc correct=True
  "partially true": ALL step1 entities correct=True  AND  step2 calc correct=False
  "false"         : ANY step1 entity has correct=False  (regardless of step2)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT CONTRACT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your program must end with exactly one print(json.dumps({...})) call that
produces this JSON schema:

{
  "step1_entities": [
    {
      "name":           "<INPUT parameter name — never the calculator output>",
      "claim_value":    "<value stated in the claim>",
      "evidence_value": "<value from evidence, or 'not found'>",
      "correct":        true or false
    }
  ],
  "step2_calculation": {
    "calculator_name": "<canonical name of the medical calculator, e.g. HAS-BLED Score>",
    "calculator_type": "<rule-based | equation-based | date-based>",
    "formula":         "<formula name and expression>",
    "substitution":    "<formula with evidence values substituted>",
    "steps":           "<arithmetic shown as a Python expression or string>",
    "correct_result":  "<your computed answer, with units>",
    "claimed_result":  "<the last claim segment value, with units>",
    "correct":         true or false
  },
  "step3_label": "<true | partially true | false>"
}

Rules:
  • Import only standard-library modules (json, math, re, datetime, etc.).
  • Do NOT import numpy, pandas, or any third-party library.
  • Do NOT read files, make network calls, or use input().
  • Output ONLY the Python code — no prose, no markdown fences.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEMPLATE A — equation-based (adapt to the actual claim and evidence):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import json

# ── STEP 1: extract and verify every INPUT entity from the evidence ─────────
entities = []

# Segment 1 — "the Calcium is 7.2 mg/dL"
claim_calcium    = 7.2        # from claim
evidence_calcium = 7.2        # from evidence
correct_calcium  = (claim_calcium == evidence_calcium)
entities.append({
    "name": "Calcium", "claim_value": "7.2 mg/dL",
    "evidence_value": "7.2 mg/dL", "correct": correct_calcium
})

# Segment 2 — "the Albumin is 3.2 g/dL"
claim_albumin    = 3.2
evidence_albumin = 3.2
correct_albumin  = (claim_albumin == evidence_albumin)
entities.append({
    "name": "Albumin", "claim_value": "3.2 g/dL",
    "evidence_value": "3.2 g/dL", "correct": correct_albumin
})

# ── STEP 2: calculate using evidence values ─────────────────────────────────
# Formula: Corrected Calcium = Calcium + 0.8 × (4.0 − Albumin)
calculator_type = "equation-based"
formula         = "Corrected Calcium (mg/dL) = Calcium + 0.8 * (4.0 - Albumin)"
substitution    = f"{evidence_calcium} + 0.8 * (4.0 - {evidence_albumin})"

correct_result_val = evidence_calcium + 0.8 * (4.0 - evidence_albumin)
claimed_result_val = 7.84     # last claim segment

# Accuracy criterion — equation-based → within 5 %
if correct_result_val != 0:
    calc_correct = abs(correct_result_val - claimed_result_val) / abs(correct_result_val) <= 0.05
else:
    calc_correct = (claimed_result_val == 0)

calculation = {
    "calculator_name": "Corrected Calcium",
    "calculator_type": calculator_type,

# ── STEP 3: derive label ────────────────────────────────────────────────────
all_step1_correct = all(e["correct"] for e in entities)
if all_step1_correct and calc_correct:
    label = "true"
elif all_step1_correct and not calc_correct:
    label = "partially true"
else:
    label = "false"

print(json.dumps({
    "step1_entities":    entities,
    "step2_calculation": calculation,
    "step3_label":       label,
}))

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEMPLATE B — rule-based (adapt to the actual claim and evidence):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import json

# ── STEP 1: extract and verify every INPUT entity from the evidence ─────────
entities = []

# Segment 1 — "the Age is 73 years"  (threshold criterion)
claim_age    = 73
evidence_age = 73       # from evidence
correct_age  = (claim_age == evidence_age)
entities.append({
    "name": "Age", "claim_value": "73 years",
    "evidence_value": "73 years", "correct": correct_age
})

# Segment 2 — "the Sex is Female"  (categorical criterion)
claim_sex    = "female"
evidence_sex = "female"  # from evidence
correct_sex  = (claim_sex.lower() == evidence_sex.lower())
entities.append({
    "name": "Sex", "claim_value": "Female",
    "evidence_value": "Female", "correct": correct_sex
})

# Segment 3 — "the Stroke history is yes"  (binary criterion)
claim_stroke    = "yes"
evidence_stroke = "no"   # from evidence — no prior stroke documented
correct_stroke  = (claim_stroke == evidence_stroke)
entities.append({
    "name": "Stroke history", "claim_value": "yes",
    "evidence_value": "no", "correct": correct_stroke
})

# ── STEP 2: calculate using evidence values ─────────────────────────────────
# Formula: CHA2DS2-VASc — evidence_* variables only, NEVER claim_* here.
calculator_type = "rule-based"
formula = "CHA2DS2-VASc Score: Age<65=0, 65-74=+1, ≥75=+2; Female=+1; Stroke=+2; ..."

score = 0

# Threshold criterion — if/elif/else blocks, NOT nested ternary
# Formula: Age <65 = 0, 65-74 = +1, ≥75 = +2
if evidence_age >= 75:
    score += 2
elif evidence_age >= 65:
    score += 1
# else: 0 points — no line needed

# Categorical criterion
score += 1 if evidence_sex.lower() == "female" else 0

# Binary criterion — evidence says "no" → 0 even though claim says "yes"
score += 2 if evidence_stroke == "yes" else 0

# … add remaining criteria in the same pattern …

correct_result_val = score
claimed_result_val = 4    # last claim segment

substitution = (
    f"Age({evidence_age}→+1) + Sex({evidence_sex}→+1) + "
    f"Stroke({evidence_stroke}→+0) + ..."
)
calc_correct = (correct_result_val == claimed_result_val)

calculation = {
    "calculator_name": "CHA2DS2-VASc Score",
    "calculator_type": calculator_type,
    "formula":         formula,
    "substitution":    substitution,
    "steps":           f"score = {correct_result_val}",
    "correct_result":  str(correct_result_val),
    "claimed_result":  str(claimed_result_val),
    "correct":         calc_correct,
}

# ── STEP 3: derive label ────────────────────────────────────────────────────
all_step1_correct = all(e["correct"] for e in entities)
if all_step1_correct and calc_correct:
    label = "true"
elif all_step1_correct and not calc_correct:
    label = "partially true"
else:
    label = "false"

print(json.dumps({
    "step1_entities":    entities,
    "step2_calculation": calculation,
    "step3_label":       label,
}))"""


# ---------------------------------------------------------------------------
# Benchmark accuracy criteria — mirrors MedCalc-Bench §3.1 Table 2
# (used by the host-side override as a safety net on top of the LLM's code)
# ---------------------------------------------------------------------------

_MODEL_EXACT_TYPES     = {'rule-based', 'date-based'}
_MODEL_TOLERANCE_TYPES = {'equation-based'}

_DATE_RE = re.compile(r'^\d{1,2}/\d{1,2}/\d{4}$')


def _infer_criteria(correct_result_str: str, model_calc_type: str = "") -> str:
    mct = (model_calc_type or "").strip().lower()
    if mct in _MODEL_EXACT_TYPES:
        return "exact"
    if mct in _MODEL_TOLERANCE_TYPES:
        return "tolerance"

    s = correct_result_str.strip()

    if _DATE_RE.match(s):
        return "exact"

    num_part = re.sub(r'[^\d.\-]', '', s).strip()
    if not num_part:
        return "exact"
    if '.' not in num_part:
        return "exact"
    return "tolerance"


def _override_step2_by_criteria(parsed: dict) -> dict:
    """Re-evaluate step2 correct flag using MedCalc-Bench criteria."""
    calc    = parsed.get("step2_calculation", {})
    cr_str  = str(calc.get("correct_result",  "")).strip()
    clm_str = str(calc.get("claimed_result", "")).strip()
    mct     = str(calc.get("calculator_type", "")).strip().lower()

    if not cr_str or not clm_str or "(unknown)" in cr_str or "(unknown)" in clm_str:
        return parsed

    criterion = _infer_criteria(cr_str, mct)

    if criterion == "exact":
        cr_core  = re.sub(r'[^\d.\-]', '', cr_str).strip()
        clm_core = re.sub(r'[^\d.\-]', '', clm_str).strip()
        if cr_core and clm_core:
            correct = (cr_core == clm_core)
        else:
            correct = (cr_str.lower() == clm_str.lower())

    else:  # tolerance
        try:
            cr_m  = re.search(r'[\-\d.]+', cr_str)
            clm_m = re.search(r'[\-\d.]+', clm_str)
            if cr_m and clm_m:
                cr_val  = float(cr_m.group())
                clm_val = float(clm_m.group())
                if cr_val == 0:
                    correct = (clm_val == 0)
                else:
                    correct = abs(cr_val - clm_val) / abs(cr_val) <= 0.05
            else:
                return parsed
        except (ValueError, TypeError):
            return parsed

    calc["correct"] = correct
    return parsed


def _remove_implicit_from_step1(parsed: dict, claim: str = "") -> dict:
    """Strip the calculator's own output from step1_entities if it slipped in."""
    entities = parsed.get("step1_entities", [])
    if not entities:
        return parsed

    output_names: set[str] = set()

    if claim:
        parts = [p.strip() for p in claim.split(",")]
        if parts:
            last_seg = parts[-1].rstrip(".")
            m = re.search(r"the\s+(.+?)\s+is\s+", last_seg, re.IGNORECASE)
            if m:
                output_names.add(m.group(1).strip().lower())

    formula = parsed.get("step2_calculation", {}).get("formula", "")
    if formula:
        formula_name = re.split(r"[=:(]", formula)[0].strip().lower()
        if formula_name:
            output_names.add(formula_name)

    if not output_names:
        return parsed

    def _is_implicit(entity_name: str) -> bool:
        n = entity_name.strip().lower()
        return any(n == out or out in n for out in output_names)

    filtered = [e for e in entities if not _is_implicit(e.get("name", ""))]
    if len(filtered) < len(entities):
        parsed["step1_entities"] = filtered

    return parsed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_label(label: str) -> str:
    label = (label or "").lower().strip()
    if label == "true":
        return "true"
    if label == "false":
        return "false"
    if "partial" in label:
        return "partially true"
    return label


def _truncate_repetition(text: str, phrase_min_len: int = 20, threshold: int = 3) -> str:
    pattern = re.compile(
        r'(.{' + str(phrase_min_len) + r',}?)\1{' + str(threshold - 1) + r',}',
        re.DOTALL
    )
    m = pattern.search(text)
    if m:
        return text[:m.start()].strip()
    return text


# ---------------------------------------------------------------------------
# Logical Reasoner (host-side; mirrors the PoT label logic as a safety net)
# ---------------------------------------------------------------------------

def _logical_reasoner(parsed: dict) -> str:
    entities = parsed.get("step1_entities", [])
    calc     = parsed.get("step2_calculation", {})

    fact1 = bool(entities) and all(bool(e.get("correct", False)) for e in entities)
    fact2 = bool(calc.get("correct", False))

    if fact1 and fact2:
        return "true"
    elif fact1 and not fact2:
        return "partially true"
    else:
        return "false"


# ---------------------------------------------------------------------------
# PoT-specific: extract Python code & execute it
# ---------------------------------------------------------------------------

def _extract_pot_code(raw_text: str) -> str:
    """
    Pull the Python program out of the raw model output.

    Handles:
      1. Properly closed markdown fences (```python ... ```)
      2. Unclosed fences
      3. Raw code with no fences (anything after stripping think blocks)
    """
    # Strip think blocks (DeepSeek-R1 / Qwen3 style)
    raw_text = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE).strip()
    raw_text = re.sub(r"^[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE).strip()
    if re.search(r"<think>", raw_text, re.IGNORECASE):
        raw_text = re.sub(r"<think>[\s\S]*$", "", raw_text, flags=re.IGNORECASE).strip()

    raw_text = _truncate_repetition(raw_text)

    # Try to pull code from a fenced block
    fence_match = re.search(
        r"```(?:python)?\s*\n([\s\S]*?)```",
        raw_text,
        re.IGNORECASE,
    )
    if fence_match:
        return fence_match.group(1).strip()

    # Unclosed fence — take everything after the opening ```
    unclosed = re.search(r"```(?:python)?\s*\n([\s\S]+)$", raw_text, re.IGNORECASE)
    if unclosed:
        return unclosed.group(1).strip()

    # No fences — return whatever is left (the model may have output raw code)
    return raw_text.strip()


# Allowed standard-library module names (single source of truth)
_ALLOWED_MODULES = (
    r"json|math|re|datetime|time|collections|functools|itertools|operator"
    r"|statistics|decimal|fractions|calendar|string|textwrap|enum|typing"
    r"|unicodedata|locale|numbers|cmath|random"
)

# Matches both styles the LLM may emit:
#   import datetime
#   from datetime import date, timedelta
_ALLOWED_IMPORTS = re.compile(
    r"^\s*(?:"
    r"import\s+(?:" + _ALLOWED_MODULES + r")\b"
    r"|from\s+(?:" + _ALLOWED_MODULES + r")(?:\.\w+)?\s+import\b"
    r")",
    re.MULTILINE,
)

_FORBIDDEN_PATTERNS = re.compile(
    r"\b(open|exec|eval|compile|__import__|importlib|subprocess|os\.|sys\."
    r"|socket|urllib|requests|http|shutil|glob|pathlib|ctypes|threading"
    r"|multiprocessing|signal|resource|pty|atexit|builtins)\b",
    re.IGNORECASE,
)


def _is_safe_code(code: str) -> tuple[bool, str]:
    """
    Lightweight safety check before exec():
      - Reject any code that references dangerous builtins or I/O.
      - Allow only a curated set of standard-library imports, accepting both
        'import X' and 'from X import Y' forms.
    Returns (is_safe, reason).
    """
    import_lines = re.findall(
        r"^\s*(?:import\s+\S+|from\s+\S+\s+import\b.*)",
        code,
        re.MULTILINE,
    )
    for line in import_lines:
        if not _ALLOWED_IMPORTS.match(line.strip()):
            return False, f"Disallowed import: {line.strip()}"

    if _FORBIDDEN_PATTERNS.search(code):
        return False, "Forbidden built-in or module reference detected."

    return True, ""


_SCORING_BUG_RE = re.compile(
    r'^(\s*score\s*\+=\s*)(\d+)(\s+if\s+.+?\s+else\s+)(\d+)(.*?)$',
    re.MULTILINE,
)


def _fix_scoring_bugs(code: str) -> str:
    """
    Detect and fix the 'else N' scoring bug before code execution.

    The LLM sometimes writes:
        score += 1 if evidence_age >= 65 else 1   # ← both branches identical
    which always adds 1 regardless of the condition — the formula threshold
    has no effect.  This happens when the LLM misreads a nested ternary in
    the template (e.g. 'else (1 if ... else 0)') as a literal 'else 1'.

    Safe fix: when the if-value and the else-value are the same integer, the
    condition is provably dead code and the else branch must be 0.

        score += N if <expr> else N   →   score += N if <expr> else 0

    Lines where the two values differ (e.g. += 2 ... else 1) are left alone
    because that is a legitimate multi-tier scoring pattern.
    """
    fixed_lines = []
    any_fixed = False

    for line in code.splitlines():
        m = _SCORING_BUG_RE.match(line)
        if m:
            prefix, val_if, mid, val_else, suffix = m.groups()
            if val_if == val_else:          # identical → dead condition → bug
                fixed_line = f"{prefix}{val_if}{mid}0{suffix}"
                print(
                    f"  [PoT] Scoring bug auto-fixed: "
                    f"'else {val_else}' → 'else 0'  |  {line.strip()}"
                )
                fixed_lines.append(fixed_line)
                any_fixed = True
                continue
        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def _execute_pot_code(code: str) -> tuple[dict | None, str]:
    """
    Execute the LLM-generated Python program in an isolated namespace.
    Returns (parsed_dict, error_str).
    On success  → (dict, "").
    On any error → (None, human-readable error string).
    """
    safe, reason = _is_safe_code(code)
    if not safe:
        msg = f"Code rejected by safety check: {reason}"
        print(f"  [PoT] {msg}")
        return None, msg

    namespace: dict = {"__builtins__": __builtins__}
    captured = io.StringIO()
    old_stdout = sys.stdout

    code = _fix_scoring_bugs(code)   # auto-correct 'else N' → 'else 0' where N==N

    try:
        sys.stdout = captured
        exec(compile(code, "<pot_program>", "exec"), namespace)  # noqa: S102
    except Exception as exc:
        sys.stdout = old_stdout
        error_str = f"{type(exc).__name__}: {exc}"
        print(f"  [PoT] Execution error: {error_str}")
        traceback.print_exc()
        return None, error_str
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue().strip()
    if not output:
        msg = "Program produced no output."
        print(f"  [PoT] {msg}")
        return None, msg

    # The last line should be valid JSON; try progressively shorter suffixes
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line), ""
            except json.JSONDecodeError:
                pass

    # Fallback: try the entire captured output
    try:
        return json.loads(output), ""
    except json.JSONDecodeError:
        msg = f"Could not parse JSON from output:\n{output[:300]}"
        print(f"  [PoT] {msg}")
        return None, msg


_MAX_REPAIR_ATTEMPTS = 2


def _repair_pot_code(
    broken_code: str,
    error: str,
    messages: list,
    model: dict,
    max_tokens: int,
    attempt: int = 1,
) -> str:
    """
    Feed the broken code and its execution error back to the LLM and ask for
    a corrected version.  Returns the raw model output (unparsed).

    The original messages (system + user) are prepended so the LLM has the
    full evidence/claim context when rewriting.  The broken code is added as
    a prior assistant turn, and the error is delivered as a new user turn.
    """
    repair_user = (
        f"Your program raised an error during execution "
        f"(repair attempt {attempt}/{_MAX_REPAIR_ATTEMPTS}):\n\n"
        f"  {error}\n\n"
        "Common causes and fixes:\n"
        "• NameError — a variable used in STEP 2 was never assigned in STEP 1,\n"
        "  or its name here does not exactly match the name defined above.\n"
        "  Use short, consistent names throughout:\n"
        "    evidence_albumin        ← define once in STEP 1\n"
        "    evidence_albumin        ← reuse the exact same name in STEP 2\n"
        "  Never create a longer derived name like evidence_albumin_corrected_anion_gap.\n"
        "• TypeError / ValueError — a numeric operation received a string.\n"
        "  Cast explicitly: float(evidence_value).\n"
        "Output ONLY the corrected Python program. No prose, no markdown fences."
    )
    repair_messages = messages + [
        {"role": "assistant", "content": broken_code},
        {"role": "user",      "content": repair_user},
    ]
    return _call_openai(model, repair_messages, max_tokens)


# ---------------------------------------------------------------------------
# Response parsing  (replaces CoT's JSON-only _parse_response)
# ---------------------------------------------------------------------------

def _keyword_label(text: str) -> str:
    """Last-resort label extraction when PoT execution fails entirely."""
    t = text.lower()
    if "partially true" in t or "partially_true" in t:
        return "partially true"
    if re.search(r'label[":\s]+false', t) or re.search(r'assign.*\bfalse\b', t):
        return "false"
    if re.search(r'label[":\s]+true', t) or re.search(r'assign.*\btrue\b', t):
        return "true"
    last_false = t.rfind('"false"')
    last_true  = t.rfind('"true"')
    if last_false > last_true:
        return "false"
    if last_true > last_false:
        return "true"
    return "false"


def _parse_response(raw_text: str, claim: str = "", repair_fn=None) -> dict:
    """
    PoT response pipeline:

      1. Extract Python code from the raw model output (strip think blocks,
         markdown fences, etc.).
      2. Safety-check the code.
      3. Execute the code; capture the printed JSON.
      4. If execution fails and repair_fn is provided, call it up to
         _MAX_REPAIR_ATTEMPTS times with (broken_code, error, attempt) and
         retry execution with the LLM-fixed code.
      5. _remove_implicit_from_step1  — strip the calculator output from step1.
      6. _override_step2_by_criteria  — enforce benchmark accuracy tolerance.
      7. _logical_reasoner            — derive the final label.
      8. Fallback: keyword scan if execution fails entirely.

    repair_fn : callable(broken_code, error_str, attempt) -> raw_text | None
        When provided, called on each execution failure so the LLM can
        self-correct naming errors, type errors, etc.

    Returns {"reasoning": str, "label": str, "pot_code": str,
             "calculator_selected": str}.
    """
    code   = _extract_pot_code(raw_text)
    parsed, error = _execute_pot_code(code) if code else (None, "no code extracted")

    # ── Self-repair loop ────────────────────────────────────────────────────
    if parsed is None and repair_fn is not None and code:
        for attempt in range(1, _MAX_REPAIR_ATTEMPTS + 1):
            print(f"  [PoT] Attempting self-repair {attempt}/{_MAX_REPAIR_ATTEMPTS}: {error}")
            repaired_raw = repair_fn(code, error, attempt)
            if not repaired_raw:
                break
            repaired_code = _extract_pot_code(repaired_raw)
            if not repaired_code:
                break
            parsed, error = _execute_pot_code(repaired_code)
            code = repaired_code          # keep the latest version for output
            if parsed is not None:
                print(f"  [PoT] Self-repair succeeded on attempt {attempt}.")
                break
        else:
            if parsed is None:
                print(f"  [PoT] Self-repair exhausted {_MAX_REPAIR_ATTEMPTS} attempts.")

    if parsed is not None:
        # Normalise alternative key names (robustness against slight schema drift)
        if not parsed.get("step1_entities"):
            for alt in ("entities", "step1", "input_entities", "extractive_claims"):
                if isinstance(parsed.get(alt), list) and parsed[alt]:
                    parsed["step1_entities"] = parsed[alt]
                    break

        if not parsed.get("step2_calculation"):
            for alt in ("calculation", "step2", "implicit_calculation", "step2_calc"):
                if isinstance(parsed.get(alt), dict) and parsed[alt]:
                    parsed["step2_calculation"] = parsed[alt]
                    break

        # If neither block is present, fall back on the raw label from the JSON
        if not parsed.get("step1_entities") and not parsed.get("step2_calculation"):
            label = normalize_label(parsed.get("step3_label", ""))
            if label not in ("true", "partially true", "false"):
                label = _keyword_label(code)
            return {"reasoning": code, "label": label, "pot_code": code}

        parsed = _remove_implicit_from_step1(parsed, claim)
        parsed = _override_step2_by_criteria(parsed)

        label = _logical_reasoner(parsed)

        # Prefer the Python-derived label; fall back to the JSON label only when
        # step1_entities is empty (e.g. the claim has no comma-separated inputs).
        if not parsed.get("step1_entities"):
            fallback = normalize_label(parsed.get("step3_label", ""))
            if fallback in ("true", "partially true", "false"):
                label = fallback

        reasoning_out = json.dumps(
            {
                "entities":      parsed.get("step1_entities", []),
                "calculation":   parsed.get("step2_calculation", {}),
                "derived_label": label,
            },
            ensure_ascii=False,
            indent=None,
        )
        calculator_selected = parsed.get("step2_calculation", {}).get("calculator_name", "")
        return {"reasoning": reasoning_out, "label": label, "pot_code": code,
                "calculator_selected": calculator_selected}

    # ── Execution failed — keyword scan over the raw model output ──────────
    return {
        "reasoning":           raw_text[:2000],
        "label":               _keyword_label(raw_text),
        "pot_code":            code,
        "calculator_selected": "",
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str) -> list:
    if not os.path.exists(csv_path):
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Evidence") and row.get("Claim") and row.get("Label"):
                rows.append({
                    "evidence":    row["Evidence"],
                    "claim":       row["Claim"],
                    "label":       normalize_label(row["Label"]),
                    "explanation": row.get("Explanation", "").strip(),
                })
    print(f"Loaded {len(rows)} samples from {csv_path}")
    return rows


# ---------------------------------------------------------------------------
# Model loading — OpenAI API client
# ---------------------------------------------------------------------------

def load_model(model_name: str) -> dict:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("WARNING: OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key or None)
    print(f"Model ready: {model_name} (OpenAI API)")
    return {"client": client, "model_name": model_name}


# ---------------------------------------------------------------------------
# Shared OpenAI call helper
# ---------------------------------------------------------------------------

def _call_openai(model: dict, messages: list, max_tokens: int) -> str:
    """
    Send a chat-completion request to the OpenAI API and return the raw text.

    Parameters
    ----------
    model      : dict returned by load_model (contains 'client' and 'model_name')
    messages   : list of {"role": ..., "content": ...} dicts
    max_tokens : maximum number of tokens to generate
    """
    client     = model["client"]
    model_name = model["model_name"]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0,          # greedy / deterministic, mirrors do_sample=False
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def _build_messages(
    item: dict,
    ev_limit: int,
    no_think: bool = False,
    formula_context: str | None = None,
) -> list:
    """
    Build the chat message list for one sample.

    formula_context : str | None
        If provided (retrieved by FormulaRAG), the formula description is
        injected into the user message as a grounding block before the
        evidence, so the LLM uses it as its authoritative formula source.

    no_think: prepends '/no_think' for Qwen3 thinking models to skip the
              <think> block and output the program directly.
    """
    formula_block = ""
    if formula_context:
        formula_block = (
            "Formula Reference (retrieved for this claim — use as the "
            "authoritative formula in STEP 2):\n"
            f"{formula_context}\n\n"
        )

    user_msg = (
        f"{formula_block}"
        f"Evidence:\n{item['evidence'][:ev_limit]}\n\n"
        f"Claim:\n{item['claim']}\n\n"
        "Output ONLY the Python program. No prose, no markdown fences."
    )
    if no_think:
        user_msg = "/no_think\n" + user_msg

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]


def _last_claim_segment(claim: str) -> str:
    """
    Extract the last comma-separated segment of the claim.

    Per the system prompt, the last segment is the implicit calculation —
    the calculator's own output (e.g. "the Corrected Calcium Concentration
    is 7.84 mg/dL").  This is the most precise signal for formula retrieval
    because it directly names the target calculator, whereas earlier segments
    are just input parameters that could belong to many different formulas.

    Example:
        "the Calcium is 7.2 mg/dL, the Albumin is 3.2 g/dL,
         the Corrected Calcium Concentration is 7.84 mg/dL."
        → "the Corrected Calcium Concentration is 7.84 mg/dL."
    """
    parts = [p.strip() for p in claim.split(",")]
    return parts[-1] if parts else claim


def _build_rag_query(item: dict, ev_chars: int = 500) -> tuple[str, str]:
    """
    Split the claim into (last_seg, earlier_segs) for weighted retrieval.

    last_seg    — the final comma-separated segment, which names the target
                  calculator directly (e.g. "the Corrected Calcium
                  Concentration is 7.84 mg/dL").  Used at full weight.

    earlier_segs — all preceding segments joined with the truncated evidence.
                   These are input parameters shared across many formulas, so
                   they are used at a reduced weight to break ties without
                   overriding the primary calculator-name signal.

    Returns
    -------
    (last_seg, earlier_segs) — both strings ready to pass to retrieve_weighted.
    """
    claim  = item.get("claim",    "").strip()
    parts  = [p.strip() for p in claim.split(",")]
    last_seg     = parts[-1] if parts else claim
    earlier_parts = parts[:-1] if len(parts) > 1 else []

    evidence     = item.get("evidence", "").strip()[:ev_chars]
    earlier_segs = " ".join(earlier_parts + ([evidence] if evidence else []))
    return last_seg, earlier_segs


# ---------------------------------------------------------------------------
# Batched inference — batch_size=1, greedy decoding
# ---------------------------------------------------------------------------

def run_batch(
    model: dict,
    batch: list,
    ev_limit: int,
    max_tokens: int,
    no_think: bool = False,
    formula_rag: "FormulaRAG | None" = None,
    rag_last_weight: float = 1.0,
    rag_earlier_weight: float = 0.3,
) -> list:
    results = []
    MIN_RAG_SCORE = 0.15

    for item in batch:
        # ── RAG: weighted retrieval over last + earlier claim segments ───────
        formula_context: str | None = None
        if formula_rag is not None:
            last_seg, earlier_segs = _build_rag_query(item)
            hits = formula_rag.retrieve_weighted(
                last_seg, earlier_segs, k=1,
                last_weight=rag_last_weight,
                earlier_weight=rag_earlier_weight,
            )
            if hits:
                formula_context, score = hits[0]
                if score < MIN_RAG_SCORE:
                    print(f"  [RAG] Score {score:.4f} below threshold — "
                          f"skipping injection, LLM will self-select formula.")
                    formula_context = None
                else:
                    print(f"  [RAG] Retrieved formula (score={score:.4f}): "
                          f"{formula_context[:80].replace(chr(10), ' ')}…")

        messages = _build_messages(
            item, ev_limit,
            no_think=no_think,
            formula_context=formula_context,
        )

        raw = _call_openai(model, messages, max_tokens)

        # Build a repair closure that captures the current prompt context
        _messages_snap   = messages
        _model_snap      = model
        _max_tokens_snap = max_tokens
        def _repair_fn(broken_code, error_str, attempt):
            return _repair_pot_code(
                broken_code, error_str,
                _messages_snap, _model_snap, _max_tokens_snap,
                attempt=attempt,
            )

        results.append(_parse_response(raw, item.get("claim", ""),
                                       repair_fn=_repair_fn))

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list):
    n = len(results)
    if n == 0:
        return

    correct = sum(1 for r in results if r["match"])
    exec_ok = sum(1 for r in results if r.get("exec_ok", False))
    print(f"\n=== Summary ===")
    print(f"Total         : {n}")
    print(f"Correct       : {correct}")
    print(f"Wrong         : {n - correct}")
    print(f"Accuracy      : {correct / n * 100:.1f}%")
    print(f"PoT exec OK   : {exec_ok}/{n}  ({exec_ok/n*100:.0f}%)")

    from collections import defaultdict
    label_stats = defaultdict(lambda: [0, 0])
    pred_dist   = defaultdict(int)
    for r in results:
        label_stats[r["true_label"]][1] += 1
        if r["match"]:
            label_stats[r["true_label"]][0] += 1
        pred_dist[r["predicted"]] += 1

    print("\nAccuracy by true label:")
    for lbl in ["true", "false", "partially true"]:
        c, t = label_stats[lbl]
        if t:
            print(f"  {lbl:<16} {c}/{t}  ({c/t*100:.0f}%)")

    print("\nPrediction distribution:")
    for lbl, cnt in sorted(pred_dist.items(), key=lambda x: -x[1]):
        print(f"  {lbl:<16} {cnt}  ({cnt/n*100:.0f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # -------------------------------------------------------------------------
    # PowerShell compatibility — fix ${VAR:-N} expansions before argparse runs.
    #
    # In Bash,  "${START:-1}"  means "use $START, or 1 if unset".
    # In PowerShell the same syntax produces an EMPTY STRING (or is dropped
    # entirely) because PowerShell does not recognise the Bash default-value
    # operator  :-  .  This causes argparse to report:
    #   "argument --start: expected one argument"
    #
    # Strategy: walk sys.argv BEFORE calling parse_args.
    #   • If a value-bearing flag is immediately followed by another flag or by
    #     an empty string, inject the canonical default so argparse never sees
    #     the missing / empty value.
    # -------------------------------------------------------------------------
    _INT_FLAG_DEFAULTS: dict[str, str] = {
        "--start":              "1",
        "--samples":            "10",
        "--ev-limit":           "2000",
        "--seed":               "42",
        "--max-tokens":         "4096",
        "--rag-k":              "1",
        "--delay":              "0",
        "--rag-last-weight":    "1.0",
        "--rag-earlier-weight": "0.3",
    }
    _patched: list[str] = []
    _raw = sys.argv[1:]
    _i = 0
    while _i < len(_raw):
        tok = _raw[_i]
        if tok in _INT_FLAG_DEFAULTS:
            _patched.append(tok)
            _i += 1
            # Peek at the next token
            if _i >= len(_raw) or _raw[_i] == "" or _raw[_i].startswith("--"):
                # Missing or empty value — substitute the default
                _patched.append(_INT_FLAG_DEFAULTS[tok])
                # Don't advance _i so the next real token is processed normally
            else:
                _patched.append(_raw[_i])
                _i += 1
        else:
            # Drop bare empty-string tokens that PowerShell may inject
            if tok != "":
                _patched.append(tok)
            _i += 1
    sys.argv[1:] = _patched

    parser = argparse.ArgumentParser(
        description=(
            "OpenAI GPT evaluator — Zero-shot Fact-Check with Program of Thought (PoT) "
            "+ Logical Reasoner.  The LLM emits Python code; the host executes it and "
            "derives the final label.  Compatible with gpt-4o-mini, gpt-4o, and any "
            "OpenAI chat model.  Requires OPENAI_API_KEY to be set in the environment."
        )
    )
    parser.add_argument("--data", required=True,
                        help="Path to CSV dataset, e.g. mrt_claim_cleaned.csv")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI model name (default: gpt-4o-mini)")
    # ---------------------------------------------------------------------------
    # PowerShell-safe int parser
    # In Bash,  ${VAR:-N}  expands to N when VAR is unset.
    # In PowerShell the same syntax expands to an EMPTY STRING, causing
    # "expected one argument" errors for integer flags.
    # _int_or_default() silently replaces "" with the flag's default value.
    # ---------------------------------------------------------------------------
    def _int_or_default(default):
        def _parse(value):
            if value == "" or value is None:
                return default
            return int(value)
        return _parse

    parser.add_argument("--samples", type=_int_or_default(10), default=10,
                        help="Number of samples to evaluate (default: 10; 0 = all)")
    parser.add_argument("--random", action="store_true",
                        help="Randomly sample instead of taking from the start")
    parser.add_argument("--ev-limit", type=_int_or_default(2000), default=2000,
                        help="Max characters of Evidence passed to model (default: 2000)")
    parser.add_argument("--seed", type=_int_or_default(42), default=42,
                        help=(
                            "Global RNG seed for Python random "
                            "(default: 42).  Governs --random sample selection."
                        ))
    parser.add_argument("--max-tokens", type=_int_or_default(4096), default=4096,
                        help=(
                            "Max output tokens per inference (default: 4096). "
                            "PoT programs are typically shorter than CoT chains, "
                            "so 2048 is usually sufficient for gpt-4o-mini."
                        ))
    parser.add_argument("--start", type=_int_or_default(1), default=1,
                        help="1-based index to start from (default: 1)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds to sleep between samples (default: 0.0)")
    parser.add_argument("--output", default="",
                        help="Output CSV path (default: auto-generated with timestamp)")
    parser.add_argument("--formula-db", default="",
                        help=(
                            "Path to the formula knowledge base "
                            "(e.g. data/web_formula.txt).  When provided, "
                            "a FormulaRAG retriever injects the closest matching "
                            "formula block into every prompt as grounding before "
                            "STEP 2.  Omit to run without RAG (original behaviour)."
                        ))
    parser.add_argument("--rag-k", type=int, default=1,
                        help="Number of formula blocks to retrieve per sample (default: 1)")
    parser.add_argument("--rag-last-weight", type=float, default=1.0,
                        help=(
                            "Retrieval weight for the last claim segment "
                            "(the implicit calculation / calculator name). "
                            "Default: 1.0"
                        ))
    parser.add_argument("--rag-earlier-weight", type=float, default=0.3,
                        help=(
                            "Retrieval weight for earlier claim segments "
                            "(input parameters — softer signal). "
                            "Default: 0.3  Set to 0 to use last segment only."
                        ))
    parser.add_argument("--no-think", action="store_true",
                        help=(
                            "No-op for OpenAI models (kept for CLI compatibility). "
                            "Originally used to prepend '/no_think' for Qwen3 models."
                        ))
    args = parser.parse_args()

    if args.random and args.start != 1:
        print(f"Warning: --start {args.start} is ignored when --random is set.")
    if args.no_think:
        print("--no-think enabled: prepending '/no_think' to all user messages.")

    # Lock down randomness
    _seed = args.seed
    random.seed(_seed)
    try:
        import numpy as np
        np.random.seed(_seed)
    except ImportError:
        pass
    print(f"Global seed set to {_seed}")

    # -- Load data -----------------------------------------------------------
    all_data = load_dataset(args.data)
    n = len(all_data) if args.samples == 0 else min(args.samples, len(all_data))

    if args.random:
        subset = random.sample(all_data, n)
        print(f"Randomly selected {n} samples (seed={_seed})")
    else:
        start_idx = args.start - 1
        subset = (all_data[start_idx:start_idx + n]
                  if args.samples != 0
                  else all_data[start_idx:])
        print(f"Using samples {args.start} to {args.start + len(subset) - 1}")

    # -- Load model ----------------------------------------------------------
    model = load_model(args.model)

    # -- Initialise FormulaRAG (optional) ------------------------------------
    formula_rag: "FormulaRAG | None" = None
    if args.formula_db:
        try:
            formula_rag = FormulaRAG(doc_path=args.formula_db)
            print(f"[RAG] Ready — {len(formula_rag)} formula blocks, "
                  f"retrieving top-{args.rag_k} per sample.\n")
        except (FileNotFoundError, ValueError) as exc:
            print(f"[RAG] WARNING: Could not load formula DB — {exc}")
            print("[RAG] Continuing without RAG grounding.\n")
    else:
        print("[RAG] No --formula-db provided; running zero-shot PoT without RAG.\n")

    # -- Prepare output CSV --------------------------------------------------
    out_path = args.output or f"results_pot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = [
        "index", "evidence", "claim", "true_label", "predicted_label",
        "match", "calculator_selected", "model_reasoning", "pot_code", "dataset_explanation",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    print(f"Writing results live to: {out_path}\n")

    # -- Evaluation loop -----------------------------------------------------
    results = []
    total   = len(subset)
    print(f"Starting PoT evaluation ({total} samples, temperature=0)...\n")

    for batch_start in range(0, total, 1):
        batch      = subset[batch_start:batch_start + 1]
        batch_idxs = [args.start + batch_start]

        print(f"[{batch_idxs[0]}/{args.start + total - 1}] Running...")

        try:
            resps = run_batch(
                model, batch, args.ev_limit,
                args.max_tokens,
                no_think=args.no_think,
                formula_rag=formula_rag,
                rag_last_weight=args.rag_last_weight,
                rag_earlier_weight=args.rag_earlier_weight,
            )
        except Exception as e:
            print(f"  Error ({e}), retrying up to 3 times...")
            resps = []
            for item in batch:
                for attempt in range(3):
                    try:
                        # Re-retrieve formula context for the retry
                        _formula_ctx: str | None = None
                        if formula_rag is not None:
                            _last, _earlier = _build_rag_query(item)
                            _hits = formula_rag.retrieve_weighted(
                                _last, _earlier, k=1,
                                last_weight=args.rag_last_weight,
                                earlier_weight=args.rag_earlier_weight,
                            )
                            if _hits and _hits[0][1] >= 0.33:
                                _formula_ctx = _hits[0][0]

                        msgs = _build_messages(
                            item, args.ev_limit,
                            no_think=args.no_think,
                            formula_context=_formula_ctx,
                        )
                        raw = _call_openai(model, msgs, args.max_tokens)
                        resps.append(_parse_response(
                            raw, item.get("claim", ""),
                            repair_fn=lambda bc, err, att: _repair_pot_code(
                                bc, err, msgs, model, args.max_tokens, attempt=att
                            ),
                        ))
                        break
                    except Exception as e2:
                        print(f"    [attempt {attempt+1}/3] Error: {e2}")
                        if attempt < 2:
                            time.sleep(30 * (attempt + 1))
                        else:
                            resps.append(
                                {"reasoning": f"Error: {e2}", "label": "false", "pot_code": ""}
                            )

        # Write results for this sample
        with open(out_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for item, idx, resp in zip(batch, batch_idxs, resps):
                predicted = normalize_label(resp["label"])
                if predicted not in ("true", "partially true", "false"):
                    predicted = "false"

                match   = predicted == item["label"]
                exec_ok = bool(resp.get("pot_code", ""))   # code was extracted
                print(f"  [{idx}] true={item['label']}  predicted={predicted}  match={match}")

                row = {
                    "index":               idx,
                    "evidence":            item["evidence"],
                    "claim":               item["claim"],
                    "true_label":          item["label"],
                    "predicted":           predicted,
                    "match":               match,
                    "exec_ok":             exec_ok,
                    "calculator_selected": resp.get("calculator_selected", ""),
                    "reasoning":           resp["reasoning"],
                    "pot_code":            resp.get("pot_code", ""),
                    "explanation":         item["explanation"],
                }
                results.append(row)

                writer.writerow({
                    "index":               row["index"],
                    "evidence":            row["evidence"],
                    "claim":               row["claim"],
                    "true_label":          row["true_label"],
                    "predicted_label":     row["predicted"],
                    "match":               "TRUE" if row["match"] else "FALSE",
                    "calculator_selected": row["calculator_selected"],
                    "model_reasoning":     row["reasoning"].replace("\n", " "),
                    "pot_code":            row["pot_code"].replace("\n", "\\n"),
                    "dataset_explanation": row["explanation"].replace("\n", " "),
                })

        if args.delay > 0 and batch_start + 1 < total:
            time.sleep(args.delay)

    print_summary(results)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()