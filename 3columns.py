import pandas as pd
import json
import re


# ---------------------------------------------------------------------------
# 1. calculator_selected
# ---------------------------------------------------------------------------

def extract_calculator_selected(formula: str) -> str:
    if not formula or formula.strip().lower() in (
        "not applicable", "not found", "", "none"
    ):
        return "not applicable"

    formula = formula.strip()
    formula = re.sub(
        r"\s*(calculation|formula)\s+not\s+provided.*",
        "",
        formula,
        flags=re.IGNORECASE,
    ).strip()

    m = re.match(r"^(.+?)\s*(?:=|:)\s", formula)
    if m:
        return m.group(1).strip()

    return formula.split("(")[0].strip()[:80]


# ---------------------------------------------------------------------------
# 2. step1_extracted_params
# ---------------------------------------------------------------------------

def parse_evidence_value(ev) -> object:
    if ev is None:
        return "not found"
    if isinstance(ev, (int, float)):
        return [float(ev)]

    ev_str = str(ev).strip()
    if not ev_str:
        return "not found"

    if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", ev_str):
        return ev_str

    bp = re.match(r"^(\d{2,3})/(\d{2,3})\s*(.*)$", ev_str)
    if bp:
        systolic  = float(bp.group(1))
        diastolic = float(bp.group(2))
        unit      = bp.group(3).strip()
        if 40 <= diastolic <= 130:
            return [systolic, unit if unit else "mmHg"]

    rate = re.match(r"^(-?[0-9]+(?:\.[0-9]+)?)\s*(/\s*\w+)\s*$", ev_str)
    if rate:
        return [float(rate.group(1)), rate.group(2).strip()]

    num_unit = re.match(r"^(-?[0-9]+(?:\.[0-9]+)?)\s+(.+)$", ev_str)
    if num_unit:
        return [float(num_unit.group(1)), num_unit.group(2).strip()]

    if re.match(r"^-?[0-9]+(?:\.[0-9]+)?$", ev_str):
        return [float(ev_str)]

    return ev_str


def extract_step1_params(entities: list) -> dict:
    params = {}
    for e in entities:
        name = e.get("name", "")
        if not name:
            continue
        params[name] = parse_evidence_value(e.get("evidence_value", "not found"))
    return params


# ---------------------------------------------------------------------------
# 3. step2_computed_value
# ---------------------------------------------------------------------------

def _days_to_gest_tuple(total_days) -> str:
    """Convert a number of days (int or float) to a gestational tuple string."""
    total_days = round(total_days)
    weeks, days = divmod(total_days, 7)
    return f"('{weeks} weeks', '{days} days')"


def _weeks_to_gest_tuple(total_weeks) -> str:
    """Convert a (possibly fractional) number of weeks to a gestational tuple string."""
    whole_weeks = int(total_weeks)
    remaining_days = round((total_weeks - whole_weeks) * 7)
    # handle rounding up to a full week
    if remaining_days == 7:
        whole_weeks += 1
        remaining_days = 0
    return f"('{whole_weeks} weeks', '{remaining_days} days')"


def _parse_gestational_age(s: str):
    """
    Try to parse s as a gestational/elapsed time expression.
    Returns a tuple string like "('X weeks', 'Y days')", or None if no match.
    Strips an optional leading 'approximately'.

    Recognised formats:
      Tuple-like  : (W weeks, D days)  |  (W, D)  |  W weeks, D days
      Timedelta   : D days, H:MM:SS
      Decimal     : X.XX weeks  |  X.XX days
      Integer     : X weeks  |  X days
      Compound    : years/months/weeks/days combinations (with optional 'approximately')
    """
    SEP            = r"\s*(?:and|,|&)?\s*"
    DAYS_PER_YEAR  = 365
    DAYS_PER_MONTH = 30
    DAYS_PER_WEEK  = 7

    # --- (W weeks, D days) unquoted tuple -----------------------------------
    m = re.match(
        r"^\(\s*(\d+)\s+weeks?\s*,\s*(\d+)\s+days?\s*\)$", s, re.IGNORECASE
    )
    if m:
        return f"('{m.group(1)} weeks', '{m.group(2)} days')"

    # --- (W, D) bare-number tuple -------------------------------------------
    m = re.match(r"^\(\s*(\d+)\s*,\s*(\d+)\s*\)$", s)
    if m:
        return f"('{m.group(1)} weeks', '{m.group(2)} days')"

    # --- Python timedelta repr: "X days, H:MM:SS" ---------------------------
    m = re.match(r"^(\d+)\s+days?,\s*\d+:\d+:\d+$", s, re.IGNORECASE)
    if m:
        return _days_to_gest_tuple(int(m.group(1)))

    # Strip optional "approximately" for all remaining patterns
    core = re.sub(r"^approximately\s+", "", s, flags=re.IGNORECASE).strip()

    # --- Decimal or integer weeks: "14.14 weeks", "7 weeks" -----------------
    m = re.match(r"^(\d+(?:\.\d+)?)\s+weeks?$", core, re.IGNORECASE)
    if m:
        return _weeks_to_gest_tuple(float(m.group(1)))

    # --- Decimal or integer days: "24.71 days", "45 days" -------------------
    m = re.match(r"^(\d+(?:\.\d+)?)\s+days?$", core, re.IGNORECASE)
    if m:
        return _days_to_gest_tuple(float(m.group(1)))

    # --- Weeks + days (no parens): "24 weeks, 5 days" -----------------------
    m = re.match(r"^(\d+)\s+weeks?" + SEP + r"(\d+)\s+days?$", core, re.IGNORECASE)
    if m:
        return f"('{m.group(1)} weeks', '{m.group(2)} days')"

    # --- Years + months + days ----------------------------------------------
    m = re.match(
        r"^(\d+)\s+years?" + SEP + r"(\d+)\s+months?" + SEP + r"(\d+)\s+days?$",
        core, re.IGNORECASE,
    )
    if m:
        total = int(m.group(1))*DAYS_PER_YEAR + int(m.group(2))*DAYS_PER_MONTH + int(m.group(3))
        return _days_to_gest_tuple(total)

    # --- Years + months -----------------------------------------------------
    m = re.match(
        r"^(\d+)\s+years?" + SEP + r"(\d+)\s+months?$", core, re.IGNORECASE,
    )
    if m:
        total = int(m.group(1))*DAYS_PER_YEAR + int(m.group(2))*DAYS_PER_MONTH
        return _days_to_gest_tuple(total)

    # --- Years only ---------------------------------------------------------
    m = re.match(r"^(\d+)\s+years?$", core, re.IGNORECASE)
    if m:
        return _days_to_gest_tuple(int(m.group(1))*DAYS_PER_YEAR)

    # --- Months + weeks + days ----------------------------------------------
    m = re.match(
        r"^(\d+)\s+months?" + SEP + r"(\d+)\s+weeks?" + SEP + r"(\d+)\s+days?$",
        core, re.IGNORECASE,
    )
    if m:
        total = int(m.group(1))*DAYS_PER_MONTH + int(m.group(2))*DAYS_PER_WEEK + int(m.group(3))
        return _days_to_gest_tuple(total)

    # --- Months + weeks -----------------------------------------------------
    m = re.match(
        r"^(\d+)\s+months?" + SEP + r"(\d+)\s+weeks?$", core, re.IGNORECASE,
    )
    if m:
        total = int(m.group(1))*DAYS_PER_MONTH + int(m.group(2))*DAYS_PER_WEEK
        return _days_to_gest_tuple(total)

    # --- Months + days ------------------------------------------------------
    m = re.match(
        r"^(\d+)\s+months?" + SEP + r"(\d+)\s+days?$", core, re.IGNORECASE,
    )
    if m:
        total = int(m.group(1))*DAYS_PER_MONTH + int(m.group(2))
        return _days_to_gest_tuple(total)

    # --- Months only --------------------------------------------------------
    m = re.match(r"^(\d+)\s+months?$", core, re.IGNORECASE)
    if m:
        return _days_to_gest_tuple(int(m.group(1))*DAYS_PER_MONTH)

    return None  # not a gestational age expression


def parse_computed_value(raw) -> object:
    """
    Return a clean representation of correct_result:
      - "YYYY-MM-DD HH:MM:SS" datetime strings    -> "MM/DD/YYYY"
      - Any gestational/elapsed time expression    -> "('X weeks', 'Y days')"
        (years/months/weeks/days; optional 'approximately'; decimal weeks/days;
         Python timedelta repr; unquoted tuple formats)
      - Number with or without units               -> float / int
      - "not applicable" / "n/a"                  -> "not applicable"
      - Missing / None                             -> "not calculated"
      - Dates MM/DD/YYYY, free text               -> kept as string
    """
    if raw is None:
        return "not calculated"

    s = str(raw).strip()

    if not s or s.lower() in ("none", ""):
        return "not calculated"

    if s.lower() in ("not applicable", "n/a"):
        return "not applicable"

    if s.lower() in (
        "not calculated",
        "not found",
        "no correct result as the evidence does not match the claim",
    ):
        return s

    # --- datetime strings "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD" ------------
    dt_match = re.match(
        r"^(\d{4})-(\d{2})-(\d{2})(?:\s+\d{2}:\d{2}:\d{2})?$", s
    )
    if dt_match:
        year, month, day = dt_match.group(1), dt_match.group(2), dt_match.group(3)
        return f"{month}/{day}/{year}"

    # --- gestational / elapsed time expressions -----------------------------
    gest = _parse_gestational_age(s)
    if gest is not None:
        return gest

    # --- date strings MM/DD/YYYY --------------------------------------------
    if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", s):
        return s

    # --- extract leading number, discarding any trailing unit ---------------
    m = re.match(r"^(-?[0-9]+(?:\.[0-9]+)?)", s)
    if m:
        val_str = m.group(1)
        return int(val_str) if "." not in val_str else float(val_str)

    return s  # fallback: return string as-is


# ---------------------------------------------------------------------------
# Helpers for parsing model_reasoning (handles both valid and truncated JSON)
# ---------------------------------------------------------------------------

def _regex_entities(text: str) -> list:
    entities = []
    for chunk in re.findall(r"\{[^{}]*?\"name\"[^{}]*?\}", text, re.DOTALL):
        try:
            entities.append(json.loads(chunk))
        except json.JSONDecodeError:
            name = re.search(r'"name":\s*"([^"]+)"', chunk)
            ev   = re.search(r'"evidence_value":\s*"([^"]*)"', chunk)
            if name and ev:
                entities.append({"name": name.group(1), "evidence_value": ev.group(1)})
    return entities


def _regex_formula(text: str):
    m = re.search(r'"formula":\s*"([^"]+)"', text)
    return m.group(1) if m else None


def _regex_correct_result(text: str):
    m = re.search(
        r'"correct_result":\s*(?:"([^"]*)"|(-?[0-9]+(?:\.[0-9]*)?))', text
    )
    if m:
        return m.group(1) if m.group(1) is not None else m.group(2)
    return None


# ---------------------------------------------------------------------------
# Main per-row extraction
# ---------------------------------------------------------------------------

def extract_from_reasoning(reasoning) -> tuple:
    if pd.isna(reasoning) or str(reasoning).strip() == "":
        return "not applicable", {}, "not calculated"

    text = str(reasoning)

    try:
        d          = json.loads(text)
        entities   = d.get("step1_entities", d.get("entities", []))
        calc       = d.get("step2_calculation", d.get("calculation", {}))
        formula    = calc.get("formula", "")
        raw_result = calc.get("correct_result", None)
    except json.JSONDecodeError:
        entities   = _regex_entities(text)
        formula    = _regex_formula(text)
        raw_result = _regex_correct_result(text)

    calculator = extract_calculator_selected(formula)
    params     = extract_step1_params(entities)
    value      = parse_computed_value(raw_result)

    return calculator, params, value


# ---------------------------------------------------------------------------
# Apply to a DataFrame
# ---------------------------------------------------------------------------

def fill_columns(df: pd.DataFrame) -> pd.DataFrame:
    calculators, params_list, values = [], [], []

    for _, row in df.iterrows():
        calc, params, value = extract_from_reasoning(row["model_reasoning"])
        calculators.append(calc)
        params_list.append(json.dumps(params))
        values.append(value)

    df = df.copy()
    df["calculator_selected"]    = calculators
    df["step1_extracted_params"] = params_list
    df["step2_computed_value"]   = values
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    INPUT  = r"02_zs_pot_llama.csv"
    OUTPUT = r"02_zs_pot_llama_full_processed.csv"

    df     = pd.read_csv(INPUT)
    df_out = fill_columns(df)
    df_out.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df_out)} rows -> {OUTPUT}")

    print("\nSample output (first 5 rows):")
    cols = ["index", "calculator_selected", "step1_extracted_params", "step2_computed_value"]
    print(df_out[cols].head().to_string(index=False))