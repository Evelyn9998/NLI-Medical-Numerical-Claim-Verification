"""
Medical Calculator Functions — 55 pre-written Python implementations.
Calculator IDs match medcalc_cal_np.csv.

Each function:
  - takes params: dict  (key names are flexible; see _get() helper)
  - returns dict: {"value": <float|int|str>, "unit": str, "note": str}
  - returns {"value": None, "unit": "", "note": "missing: <param>"} on error
"""

import math
from datetime import date, timedelta
from typing import Any, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(params: dict, *keys, default=None, cast=float):
    """Try multiple key names (case-insensitive, normalised).  Return first match."""
    def norm(s):
        return s.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    normed = {norm(k): v for k, v in params.items()}
    for key in keys:
        v = normed.get(norm(key))
        if v is not None:
            if cast is None:
                return v
            try:
                return cast(v)
            except (ValueError, TypeError):
                pass
    return default


def _female(params: dict) -> Optional[bool]:
    v = _get(params, "sex", "gender", cast=str)
    if v is None:
        return None
    return "f" in v.lower()


def _black(params: dict) -> bool:
    v = _get(params, "race", "ethnicity", cast=str)
    if v is None:
        return False
    return "black" in v.lower() or "african" in v.lower()


def _ok(value, unit="", note=""):
    return {"value": value, "unit": unit, "note": note}


def _missing(*names):
    return {"value": None, "unit": "", "note": "missing: " + ", ".join(names)}


# ── Unit auto-detection ───────────────────────────────────────────────────────

def _cr_mgdl(v) -> Optional[float]:
    """Creatinine: if value > 20 it is likely µmol/L → convert to mg/dL (÷88.4)."""
    if v is None:
        return None
    f = float(v)
    return round(f / 88.4, 4) if f > 20 else f


def _glucose_mgdl(v) -> Optional[float]:
    """Glucose: if value < 25 it is likely mmol/L → convert to mg/dL (×18)."""
    if v is None:
        return None
    f = float(v)
    return round(f * 18.0, 2) if f < 25 else f


def _plt_k(v) -> Optional[float]:
    """Platelets or WBC: if value > 1000 it is /µL → convert to ×10³/µL (÷1000)."""
    if v is None:
        return None
    f = float(v)
    return round(f / 1000.0, 2) if f > 1000 else f


def _temp_c(v) -> Optional[float]:
    """Temperature: if value > 44 it is °F → convert to °C."""
    if v is None:
        return None
    f = float(v)
    return round((f - 32) * 5 / 9, 2) if f > 44 else f


def _height_cm(v) -> Optional[float]:
    """Height: if value < 3 it is likely in metres → convert to cm (×100)."""
    if v is None:
        return None
    f = float(v)
    return round(f * 100, 2) if f < 3 else f


def _albumin_gdl(v) -> Optional[float]:
    """Albumin: if value > 10 it is likely in g/L → convert to g/dL (÷10)."""
    if v is None:
        return None
    f = float(v)
    return round(f / 10.0, 3) if f > 10 else f


def _chol_mgdl(v) -> Optional[float]:
    """Cholesterol/TG: if value < 20 it is likely mmol/L → convert to mg/dL (×38.67)."""
    if v is None:
        return None
    f = float(v)
    return round(f * 38.67, 2) if f < 20 else f


# ─────────────────────────────────────────────────────────────────────────────
# ID 2 — Creatinine Clearance (Cockcroft-Gault)
# ─────────────────────────────────────────────────────────────────────────────

def calc_2(params: dict) -> dict:
    age = _get(params, "age")
    wt  = _get(params, "weight", "weight_kg", "actual_body_weight")
    cr  = _cr_mgdl(_get(params, "creatinine", "serum_creatinine", "scr", "cr"))
    sex = _female(params)
    ht  = _height_cm(_get(params, "height", "height_cm"))
    if None in (age, wt, cr, sex):
        return _missing("age, weight, creatinine, sex")
    # Use adjusted body weight if actual weight > IBW and height is provided
    effective_wt = wt
    if ht is not None:
        ht_in = ht / 2.54
        ibw = (45.5 if sex else 50.0) + 2.3 * (ht_in - 60)
        if wt > ibw > 0:
            effective_wt = ibw + 0.4 * (wt - ibw)
    crcl = ((140 - age) * effective_wt) / (72 * cr)
    if sex:
        crcl *= 0.85
    return _ok(round(crcl, 5), "mL/min")


# ─────────────────────────────────────────────────────────────────────────────
# ID 3 — CKD-EPI GFR (2009 equation)
# ─────────────────────────────────────────────────────────────────────────────

def calc_3(params: dict) -> dict:
    """CKD-EPI 2021 (race-free) equation."""
    age = _get(params, "age")
    cr  = _cr_mgdl(_get(params, "creatinine", "serum_creatinine", "scr"))
    sex = _female(params)
    if None in (age, cr, sex):
        return _missing("age, creatinine, sex")
    kappa = 0.7  if sex else 0.9
    alpha = -0.241 if sex else -0.302
    ratio = cr / kappa
    gfr = (142
           * (min(ratio, 1) ** alpha)
           * (max(ratio, 1) ** -1.200)
           * (0.9938 ** age))
    if sex:
        gfr *= 1.012
    return _ok(round(gfr, 4), "mL/min/1.73 m²")


# ─────────────────────────────────────────────────────────────────────────────
# ID 4 — CHA₂DS₂-VASc
# ─────────────────────────────────────────────────────────────────────────────

def calc_4(params: dict) -> dict:
    age = _get(params, "age")
    sex = _female(params)
    if age is None:
        return _missing("age")
    score = 0
    score += int(bool(_get(params, "congestive_heart_failure", "chf",
                           "congestive heart failure", default=0, cast=int)))
    score += int(bool(_get(params, "hypertension", "hypertension_history",
                           "hypertension history", default=0, cast=int)))
    score += 2 if age >= 75 else (1 if age >= 65 else 0)
    score += int(bool(_get(params, "diabetes", "diabetes_history",
                           "diabetes mellitus", default=0, cast=int)))
    score += 2 * int(bool(_get(params, "stroke", "thromboembolism",
                               "stroke_history", "tia",
                               "transient_ischemic_attacks_history",
                               default=0, cast=int)))
    score += int(bool(_get(params, "vascular_disease",
                           "vascular disease history", default=0, cast=int)))
    if sex:  # female
        score += 1
    return _ok(score, "points")


# ─────────────────────────────────────────────────────────────────────────────
# ID 5 — Mean Arterial Pressure (MAP)
# ─────────────────────────────────────────────────────────────────────────────

def calc_5(params: dict) -> dict:
    sbp = _get(params, "sbp", "systolic_blood_pressure", "systolic blood pressure")
    dbp = _get(params, "dbp", "diastolic_blood_pressure", "diastolic blood pressure")
    if None in (sbp, dbp):
        return _missing("sbp, dbp")
    return _ok(round((sbp + 2 * dbp) / 3, 4), "mm Hg")


# ─────────────────────────────────────────────────────────────────────────────
# ID 6 — Body Mass Index (BMI)
# ─────────────────────────────────────────────────────────────────────────────

def calc_6(params: dict) -> dict:
    wt = _get(params, "weight", "weight_kg")
    ht = _height_cm(_get(params, "height", "height_cm"))
    if None in (wt, ht):
        return _missing("weight, height")
    bmi = wt / (ht / 100) ** 2
    return _ok(round(bmi, 5), "kg/m²")


# ─────────────────────────────────────────────────────────────────────────────
# ID 7 — Calcium Correction for Hypoalbuminemia
# ─────────────────────────────────────────────────────────────────────────────

def calc_7(params: dict) -> dict:
    ca  = _get(params, "calcium", "ca", "serum_calcium")
    alb = _albumin_gdl(_get(params, "albumin", "alb", "serum_albumin"))
    if None in (ca, alb):
        return _missing("calcium, albumin")
    corrected = ca + 0.8 * (4.0 - alb)
    return _ok(round(corrected, 4), "mg/dL")


# ─────────────────────────────────────────────────────────────────────────────
# ID 8 — Wells' Criteria for Pulmonary Embolism
# ─────────────────────────────────────────────────────────────────────────────

def calc_8(params: dict) -> dict:
    def flag(key, *aliases):
        return float(bool(_get(params, key, *aliases, default=0, cast=int)))

    hr   = _get(params, "heart_rate", "hr", "heart rate or pulse", default=0)
    dvt  = flag("clinical_signs_dvt",
                "clinical signs and symptoms of deep vein thrombosis",
                "clinical_signs_and_symptoms_of_deep_vein_thrombosis")
    pe1  = flag("pe_top_diagnosis",
                "pulmonary embolism is #1 diagnosis or equally likely",
                "pe_most_likely_diagnosis")
    hr_f = 1.5 if hr > 100 else 0
    # immobilisation ≥3 days OR surgery in previous 4 weeks → same +1.5 criterion
    immob = 1.5 * int(bool(
        flag("immobilization", "immobilization_for_at_least_3_days",
             "immobilization for at least 3 days")
        or flag("surgery_previous_4_weeks",
                "surgery in the previous 4 weeks",
                "surgery_in_the_previous_4_weeks")
    ))
    # prior PE OR prior DVT → same +1.5 criterion (check each separately, OR together)
    prev = 1.5 * int(bool(
        flag("previously documented pulmonary embolism",
             "previously_documented_pulmonary_embolism",
             "previously documented pe")
        or flag("previously documented deep vein thrombosis",
                "previously_documented_deep_vein_thrombosis",
                "previously documented dvt")
    ))
    hemo  = flag("hemoptysis")
    malig = flag("malignancy",
                 "malignancy with treatment within 6 months or palliative")
    score = dvt * 3 + pe1 * 3 + hr_f + immob + prev + hemo + malig
    return _ok(score, "points")


# ─────────────────────────────────────────────────────────────────────────────
# ID 9 — MDRD GFR Equation
# ─────────────────────────────────────────────────────────────────────────────

def calc_9(params: dict) -> dict:
    age = _get(params, "age")
    cr  = _cr_mgdl(_get(params, "creatinine", "serum_creatinine", "scr"))
    sex = _female(params)
    if None in (age, cr, sex):
        return _missing("age, creatinine, sex")
    # 175 = IDMS-traceable coefficient (MedCalc standard); legacy formula used 186
    gfr = 175 * (cr ** -1.154) * (age ** -0.203)
    if sex:
        gfr *= 0.742
    if _black(params):
        gfr *= 1.212
    return _ok(round(gfr, 4), "mL/min/1.73 m²")


# ─────────────────────────────────────────────────────────────────────────────
# ID 10 — Ideal Body Weight (Devine formula)
# ─────────────────────────────────────────────────────────────────────────────

def calc_10(params: dict) -> dict:
    ht  = _height_cm(_get(params, "height", "height_cm"))
    sex = _female(params)
    if None in (ht, sex):
        return _missing("height, sex")
    ht_in = ht / 2.54
    base = 45.5 if sex else 50.0
    ibw  = base + 2.3 * (ht_in - 60)
    return _ok(round(ibw, 4), "kg")


# ─────────────────────────────────────────────────────────────────────────────
# ID 11 — QTc Bazett
# ─────────────────────────────────────────────────────────────────────────────

def calc_11(params: dict) -> dict:
    qt = _get(params, "qt_interval", "qt", "qt interval")
    hr = _get(params, "heart_rate", "hr", "heart rate or pulse",
              "heart_rate_or_pulse")
    if None in (qt, hr):
        return _missing("qt_interval, heart_rate")
    rr = 60 / hr  # seconds
    qtc = qt / math.sqrt(rr)
    return _ok(round(qtc, 4), "ms")


# ─────────────────────────────────────────────────────────────────────────────
# ID 13 — Estimated Due Date (Naegele's rule)
# ─────────────────────────────────────────────────────────────────────────────

def calc_13(params: dict) -> dict:
    lmp_raw = _get(params, "last_menstrual_date", "lmp",
                   "last_menstrual_period", cast=str)
    cycle   = _get(params, "cycle_length", "cycle", default=28)
    if lmp_raw is None:
        return _missing("last_menstrual_date")
    try:
        lmp = date.fromisoformat(lmp_raw.strip())
    except ValueError:
        return {"value": None, "unit": "", "note": f"cannot parse date: {lmp_raw}"}
    edd = lmp + timedelta(days=280 - 28 + cycle)
    return _ok(edd.isoformat(), "date")


# ─────────────────────────────────────────────────────────────────────────────
# ID 15 — Child-Pugh Score for Cirrhosis Mortality
# ─────────────────────────────────────────────────────────────────────────────

def calc_15(params: dict) -> dict:
    alb   = _albumin_gdl(_get(params, "albumin", "alb"))
    bili  = _get(params, "bilirubin", "total_bilirubin")
    inr   = _get(params, "inr", "international_normalized_ratio",
                 "international normalized ratio")
    asc   = _get(params, "ascites", cast=str)
    enc   = _get(params, "encephalopathy", cast=str)
    if None in (alb, bili, inr, asc, enc):
        return _missing("albumin, bilirubin, inr, ascites, encephalopathy")

    # Albumin
    if alb > 3.5:    alb_pts = 1
    elif alb >= 2.8: alb_pts = 2
    else:            alb_pts = 3

    # Bilirubin
    if bili < 2:    bili_pts = 1
    elif bili <= 3: bili_pts = 2
    else:           bili_pts = 3

    # INR
    if inr < 1.7:    inr_pts = 1
    elif inr <= 2.3: inr_pts = 2
    else:            inr_pts = 3

    # Ascites
    asc = asc.lower()
    if "none" in asc or "absent" in asc or "no " in asc or asc in ("0","no"):
        asc_pts = 1
    elif "slight" in asc or "mild" in asc or "small" in asc or "1" in asc:
        asc_pts = 2
    else:
        asc_pts = 3

    # Encephalopathy
    enc = enc.lower()
    if "none" in enc or "no " in enc or enc in ("0","no"):
        enc_pts = 1
    elif "grade 1" in enc or "grade 2" in enc or "1-2" in enc:
        enc_pts = 2
    else:
        enc_pts = 3

    score = alb_pts + bili_pts + inr_pts + asc_pts + enc_pts
    cls = "A" if score <= 6 else ("B" if score <= 9 else "C")
    return _ok(score, "points", note=f"Class {cls}")


# ─────────────────────────────────────────────────────────────────────────────
# ID 16 — Wells' Criteria for DVT
# ─────────────────────────────────────────────────────────────────────────────

def calc_16(params: dict) -> dict:
    def flag(*keys):
        return int(bool(_get(params, *keys, default=0, cast=int)))

    score = (
        flag("active_cancer", "active cancer")
        + flag("paralysis_paresis_plaster",
               "paralysis, paresis, or recent plaster immobilization of the lower extremity")
        + flag("bedridden_recently", "bedridden recently >3 days",
               "bedridden_recently_3_days")
        + flag("major_surgery", "major surgery within 12 weeks")
        + flag("localized_tenderness",
               "localized tenderness along the deep venous system")
        + flag("entire_leg_swollen", "entire leg swollen")
        + flag("calf_swelling",
               "calf swelling >3 centimeters compared to the other leg")
        + flag("pitting_edema",
               "pitting edema, confined to symptomatic leg")
        + flag("collateral_veins",
               "collateral (nonvaricose) superficial veins present",
               "collateral_superficial_veins_present")
        + flag("previous_dvt",
               "previously documented deep vein thrombosis")
        - 2 * flag("alternative_diagnosis",
                   "alternative diagnosis to deep vein thrombosis as likely or more likely",
                   "alternative_diagnosis_as_likely_or_more_likely")
    )
    return _ok(score, "points")


# ─────────────────────────────────────────────────────────────────────────────
# ID 17 — Revised Cardiac Risk Index (RCRI / Lee Score)
# ─────────────────────────────────────────────────────────────────────────────

def calc_17(params: dict) -> dict:
    def flag(*keys):
        return int(bool(_get(params, *keys, default=0, cast=int)))

    cr  = _get(params, "creatinine", "preoperative_creatinine",
               "pre_operative_creatinine", default=0)
    score = (
        flag("high_risk_surgery", "elevated_risk_surgery", "elevated-risk surgery")
        + flag("ischemic_heart_disease",
               "history of ischemic heart disease")
        + flag("chf", "congestive_heart_failure",
               "congestive heart failure criteria for the cardiac risk index rule")
        + flag("cerebrovascular_disease",
               "history of cerebrovascular disease")
        + flag("insulin", "preoperative_treatment_with_insulin",
               "pre-operative treatment with insulin")
        + (1 if cr > 2.0 else 0)
    )
    return _ok(score, "points")


# ─────────────────────────────────────────────────────────────────────────────
# ID 18 — HEART Score
# ─────────────────────────────────────────────────────────────────────────────

def calc_18(params: dict) -> dict:
    history   = _get(params, "history", "suspicion_history",
                     "suspicion history", default=0, cast=int)
    ecg       = _get(params, "ecg", "electrocardiogram",
                     "electrocardiogram test", default=0, cast=int)
    age       = _get(params, "age")
    risk      = _get(params, "risk_factors", "atherosclerotic_disease",
                     default=0, cast=int)
    troponin  = _get(params, "troponin", "initial_troponin",
                     "initial troponin", default=0, cast=int)

    if age is None:
        return _missing("age")

    age_pts = 2 if age >= 65 else (1 if age >= 45 else 0)
    score   = history + ecg + age_pts + risk + troponin
    return _ok(score, "points")


# ─────────────────────────────────────────────────────────────────────────────
# ID 19 — FIB-4 Index for Liver Fibrosis
# ─────────────────────────────────────────────────────────────────────────────

def calc_19(params: dict) -> dict:
    age  = _get(params, "age")
    ast  = _get(params, "ast", "aspartate_aminotransferase",
                "aspartate aminotransferase")
    alt  = _get(params, "alt", "alanine_aminotransferase",
                "alanine aminotransferase")
    plt  = _plt_k(_get(params, "platelets", "platelet_count",
                       "platelet count"))
    if None in (age, ast, alt, plt):
        return _missing("age, ast, alt, platelets")
    fib4 = (age * ast) / (plt * math.sqrt(alt))
    return _ok(round(fib4, 4), "")


# ─────────────────────────────────────────────────────────────────────────────
# ID 20 — Centor Score (Modified/McIsaac) for Strep Pharyngitis
# ─────────────────────────────────────────────────────────────────────────────

def calc_20(params: dict) -> dict:
    age  = _get(params, "age")
    temp = _temp_c(_get(params, "temperature", "temp", default=0))
    cough_absent = _get(params, "cough_absent", "absence_of_cough",
                        "cough absent", default=0, cast=int)
    exudate      = _get(params, "exudate", "tonsillar_exudate",
                        "exudate or swelling on tonsils",
                        default=0, cast=int)
    lymph        = _get(params, "lymph_nodes",
                        "tender/swollen anterior cervical lymph nodes",
                        "tender_swollen_anterior_cervical_lymph_nodes",
                        default=0, cast=int)
    if age is None:
        return _missing("age")

    score = 0
    if temp > 38:
        score += 1
    score += int(bool(cough_absent))
    score += int(bool(exudate))
    score += int(bool(lymph))
    # Age modifier
    if age < 15:
        score += 1
    elif age >= 45:
        score -= 1
    return _ok(score, "points")


# ─────────────────────────────────────────────────────────────────────────────
# ID 21 — Glasgow Coma Score (GCS)
# ─────────────────────────────────────────────────────────────────────────────

_GCS_EYE = {
    "no eye opening": 1, "no_eye_opening": 1,
    "to pain": 2, "eye opening to pain": 2, "to_pain": 2,
    "to verbal command": 3, "eye opening to verbal command": 3, "to verbal": 3, "to_verbal_command": 3,
    "spontaneously": 4, "spontaneous": 4, "eyes open spontaneously": 4,
}
_GCS_VERBAL = {
    "no verbal response": 1, "no response": 1, "none": 1,
    "incomprehensible sounds": 2, "incomprehensible": 2, "sounds": 2,
    "inappropriate words": 3, "inappropriate": 3,
    "confused": 4, "disoriented": 4,
    "oriented": 5, "normal conversation": 5,
}
_GCS_MOTOR = {
    "no motor response": 1, "no response": 1, "none": 1,
    "extension to pain": 2, "extensor response": 2, "abnormal extension": 2, "decerebrate": 2,
    "flexion to pain": 3, "abnormal flexion": 3, "decorticate": 3,
    "withdrawal from pain": 4, "withdrawal": 4,
    "localizes pain": 5, "localizes": 5,
    "obeys commands": 6, "follows commands": 6, "obeys": 6,
}


def _gcs_lookup(table, val):
    if isinstance(val, (int, float)):
        return int(val)
    v = str(val).lower().strip()
    for k, s in table.items():
        if k in v or v in k:
            return s
    return None


def calc_21(params: dict) -> dict:
    eye   = _get(params, "best_eye_response", "eye", "eye_response",
                 "best eye response", cast=None)
    verb  = _get(params, "best_verbal_response", "verbal",
                 "best verbal response", cast=None)
    motor = _get(params, "best_motor_response", "motor",
                 "best motor response", cast=None)
    if None in (eye, verb, motor):
        return _missing("eye, verbal, motor response")
    e = _gcs_lookup(_GCS_EYE,   eye)
    v = _gcs_lookup(_GCS_VERBAL, verb)
    m = _gcs_lookup(_GCS_MOTOR,  motor)
    if None in (e, v, m):
        return {"value": None, "unit": "points",
                "note": f"unrecognised GCS descriptor(s): eye={eye}, verbal={verb}, motor={motor}"}
    return _ok(e + v + m, "points")


# ─────────────────────────────────────────────────────────────────────────────
# ID 22 — Maintenance Fluids (Holliday-Segar)
# ─────────────────────────────────────────────────────────────────────────────

def calc_22(params: dict) -> dict:
    """Holliday-Segar 4-2-1 rule → mL/hr (MDCalc standard output)."""
    wt = _get(params, "weight", "weight_kg")
    if wt is None:
        return _missing("weight")
    if wt <= 10:
        rate = wt * 4
    elif wt <= 20:
        rate = 40 + (wt - 10) * 2
    else:
        rate = 60 + (wt - 20) * 1
    return _ok(round(rate, 2), "mL/hr")


# ─────────────────────────────────────────────────────────────────────────────
# ID 23 — MELD Na (UNOS/OPTN)
# ─────────────────────────────────────────────────────────────────────────────

def calc_23(params: dict) -> dict:
    bili = _get(params, "bilirubin", "total_bilirubin")
    cr   = _cr_mgdl(_get(params, "creatinine", "serum_creatinine", "scr"))
    inr  = _get(params, "inr", "international_normalized_ratio",
                "international normalized ratio")
    na   = _get(params, "sodium", "serum_sodium", "na")
    dialysis = _get(params, "dialysis",
                    "dialysis at least twice in the past week",
                    "continuous_veno_venous_hemodialysis",
                    default=0, cast=int)
    if None in (bili, cr, inr, na):
        return _missing("bilirubin, creatinine, inr, sodium")

    if dialysis:
        cr = 4.0
    cr   = max(1.0, min(cr,   4.0))
    bili = max(1.0, bili)
    inr  = max(1.0, inr)
    na   = max(125, min(na, 137))

    meld = 3.78 * math.log(bili) + 11.2 * math.log(inr) + 9.57 * math.log(cr) + 6.43
    meld = round(meld)
    meld_na = meld + 1.32 * (137 - na) - 0.033 * meld * (137 - na)
    return _ok(round(meld_na, 4), "points")


# ─────────────────────────────────────────────────────────────────────────────
# ID 24 — Steroid Conversion Calculator
# ─────────────────────────────────────────────────────────────────────────────

_STEROID_EQ = {
    "hydrocortisone": 20, "cortisol": 20,
    "cortisone": 25,
    "prednisone": 5,
    "prednisolone": 5,
    "methylprednisolone": 4, "methyl prednisolone": 4,
    "triamcinolone": 4,
    "dexamethasone": 0.75,
    "betamethasone": 0.6,
    "budesonide": 3,
    "fluticasone": 2,
    "fludrocortisone": 10,
}

def calc_24(params: dict) -> dict:
    drug_in  = _get(params, "input_steroid", "from_steroid", "steroid", cast=str)
    dose_in  = _get(params, "dose", "input_dose", "dose_mg")
    drug_out = _get(params, "target_steroid", "to_steroid", cast=str)
    if None in (drug_in, dose_in, drug_out):
        return _missing("input_steroid, dose, target_steroid")
    eq_in  = next((v for k, v in _STEROID_EQ.items() if k in drug_in.lower()),  None)
    eq_out = next((v for k, v in _STEROID_EQ.items() if k in drug_out.lower()), None)
    if None in (eq_in, eq_out):
        return {"value": None, "unit": "mg",
                "note": f"unknown steroid: {drug_in} or {drug_out}"}
    equiv = dose_in / eq_in * eq_out
    return _ok(round(equiv, 4), "mg")


# ─────────────────────────────────────────────────────────────────────────────
# ID 25 — HAS-BLED Score
# ─────────────────────────────────────────────────────────────────────────────

def calc_25(params: dict) -> dict:
    def flag(*keys):
        return int(bool(_get(params, *keys, default=0, cast=int)))

    age = _get(params, "age", default=0)
    sbp = _get(params, "sbp", "systolic_blood_pressure",
               "systolic blood pressure", default=0)
    score = (
        (1 if sbp > 160 else flag("hypertension"))
        + flag("renal_disease",
               "renal disease criteria for the has-bled rule")
        + flag("liver_disease",
               "liver disease criteria for the has-bled rule")
        + flag("stroke", "stroke_history")
        + flag("prior_bleeding",
               "prior major bleeding or predisposition to bleeding")
        + flag("labile_inr",
               "labile international normalized ratio",
               "labile_international_normalized_ratio")
        + (1 if age > 65 else 0)
        + flag("medication", "medication usage predisposing to bleeding",
               "drugs")
        + flag("alcohol",
               "number of alcoholic drinks per week",
               "alcohol_use")
    )
    return _ok(score, "points")


# ─────────────────────────────────────────────────────────────────────────────
# ID 26 — Sodium Correction for Hyperglycemia
# ─────────────────────────────────────────────────────────────────────────────

def calc_26(params: dict) -> dict:
    """Hillier 1999 formula (MedCalc default): +0.024 mEq/L per mg/dL glucose above 100."""
    na  = _get(params, "sodium", "serum_sodium", "na")
    glu = _glucose_mgdl(_get(params, "glucose", "serum_glucose", "blood_glucose"))
    if None in (na, glu):
        return _missing("sodium, glucose")
    corrected = na + 0.024 * (glu - 100)
    return _ok(round(corrected, 4), "mEq/L")


# ─────────────────────────────────────────────────────────────────────────────
# ID 27 — Glasgow-Blatchford Bleeding Score (GBS)
# ─────────────────────────────────────────────────────────────────────────────

def calc_27(params: dict) -> dict:
    bun  = _get(params, "bun", "blood_urea_nitrogen",
                "blood urea nitrogen (bun)")
    hgb  = _get(params, "hemoglobin", "hgb", "hb")
    sbp  = _get(params, "sbp", "systolic_blood_pressure",
                "systolic blood pressure")
    hr   = _get(params, "hr", "heart_rate", "heart rate or pulse",
                "heart_rate_or_pulse", default=0)
    sex  = _female(params)
    if None in (bun, hgb, sbp, sex):
        return _missing("bun, hemoglobin, sbp, sex")

    # BUN points (mg/dL)
    if   bun >= 70:   bun_pts = 6
    elif bun >= 28:   bun_pts = 4
    elif bun >= 22.4: bun_pts = 3
    elif bun >= 18.2: bun_pts = 2
    else:             bun_pts = 0

    # Hemoglobin points
    if sex:  # female
        if   hgb < 10:  hgb_pts = 6
        elif hgb < 12:  hgb_pts = 1
        else:           hgb_pts = 0
    else:    # male
        if   hgb < 10:  hgb_pts = 6
        elif hgb < 12:  hgb_pts = 3
        elif hgb < 13:  hgb_pts = 1
        else:           hgb_pts = 0

    # SBP points
    if   sbp < 90:   sbp_pts = 3
    elif sbp < 100:  sbp_pts = 2
    elif sbp < 109:  sbp_pts = 1
    else:            sbp_pts = 0

    other = (
        (1 if hr >= 100 else 0)
        + int(bool(_get(params, "melena", "melena_present",
                        "melena present", default=0, cast=int)))
        + 2 * int(bool(_get(params, "syncope", "recent_syncope",
                             "recent syncope", default=0, cast=int)))
        + 2 * int(bool(_get(params, "hepatic_disease",
                             "hepatic disease history", default=0, cast=int)))
        + 2 * int(bool(_get(params, "cardiac_failure",
                             "cardiac failure present", default=0, cast=int)))
    )
    return _ok(bun_pts + hgb_pts + sbp_pts + other, "points")


# ─────────────────────────────────────────────────────────────────────────────
# ID 28 — APACHE II Score
# ─────────────────────────────────────────────────────────────────────────────

def _aps_score(val, thresholds):
    """thresholds: list of (lower_bound, upper_bound, score) from worst-low to normal to worst-high"""
    if val is None:
        return 0
    for lo, hi, pts in thresholds:
        if lo <= val < hi:
            return pts
    return 0


def calc_28(params: dict) -> dict:
    temp  = _temp_c(_get(params, "temperature", "temp", default=None))
    map_  = _get(params, "map", "mean_arterial_pressure", default=None)
    if map_ is None:
        sbp = _get(params, "sbp", "systolic_blood_pressure", default=None)
        dbp = _get(params, "dbp", "diastolic_blood_pressure", default=None)
        if sbp is not None and dbp is not None:
            map_ = (sbp + 2 * dbp) / 3
    hr    = _get(params, "hr", "heart_rate", "heart rate or pulse", default=None)
    rr    = _get(params, "rr", "respiratory_rate", default=None)
    pao2  = _get(params, "pao2", default=None)
    fio2  = _get(params, "fio2", default=None)
    aa    = _get(params, "aa_gradient", "a_a_gradient", default=None)
    ph    = _get(params, "ph", "arterial_ph", default=None)
    na    = _get(params, "sodium", "serum_sodium", default=None)
    k     = _get(params, "potassium", "serum_potassium", default=None)
    cr    = _cr_mgdl(_get(params, "creatinine", "serum_creatinine", default=None))
    hct   = _get(params, "hematocrit", "hct", default=None)
    wbc   = _plt_k(_get(params, "wbc", "white_blood_cell_count",
                        "white blood cell count", default=None))
    gcs   = _get(params, "gcs", "glasgow_coma_score",
                 "glasgow coma score", default=15)
    age   = _get(params, "age")
    arf   = _get(params, "acute_renal_failure",
                 "acute renal failure", default=0, cast=int)
    chronic = _get(params, "chronic_health",
                   "history of severe organ failure or immunocompromise",
                   default=0, cast=int)
    surgery_type = _get(params, "surgery_type",
                        "surgery type", default="none", cast=str)

    if age is None:
        return _missing("age")

    score = 0

    # Temperature
    if temp is not None:
        if   temp >= 41:   score += 4
        elif temp >= 39:   score += 3
        elif temp >= 38.5: score += 1
        elif temp >= 36:   score += 0
        elif temp >= 34:   score += 1
        elif temp >= 32:   score += 2
        elif temp >= 30:   score += 3
        else:              score += 4

    # MAP
    if map_ is not None:
        if   map_ >= 160: score += 4
        elif map_ >= 130: score += 3
        elif map_ >= 110: score += 2
        elif map_ >= 70:  score += 0
        elif map_ >= 50:  score += 2
        else:             score += 4

    # HR
    if hr is not None:
        if   hr >= 180: score += 4
        elif hr >= 140: score += 3
        elif hr >= 110: score += 2
        elif hr >= 70:  score += 0
        elif hr >= 55:  score += 2
        elif hr >= 40:  score += 3
        else:           score += 4

    # RR
    if rr is not None:
        if   rr >= 50: score += 4
        elif rr >= 35: score += 3
        elif rr >= 25: score += 1
        elif rr >= 12: score += 0
        elif rr >= 10: score += 1
        elif rr >= 6:  score += 2
        else:          score += 4

    # Oxygenation
    if fio2 is not None and fio2 >= 0.5:
        if aa is None and pao2 is not None:
            aa = (fio2 * 713) - pao2  # simplified A-a gradient
        if aa is not None:
            if   aa >= 500: score += 4
            elif aa >= 350: score += 3
            elif aa >= 200: score += 2
            else:           score += 0
    elif pao2 is not None:
        if   pao2 < 55:  score += 4
        elif pao2 < 60:  score += 3
        elif pao2 < 70:  score += 1
        else:            score += 0

    # pH
    if ph is not None:
        if   ph >= 7.7:   score += 4
        elif ph >= 7.6:   score += 3
        elif ph >= 7.5:   score += 1
        elif ph >= 7.33:  score += 0
        elif ph >= 7.25:  score += 2
        elif ph >= 7.15:  score += 3
        else:             score += 4

    # Sodium
    if na is not None:
        if   na >= 180: score += 4
        elif na >= 160: score += 3
        elif na >= 155: score += 2
        elif na >= 150: score += 1
        elif na >= 130: score += 0
        elif na >= 120: score += 2
        elif na >= 111: score += 3
        else:           score += 4

    # Potassium
    if k is not None:
        if   k >= 7:   score += 4
        elif k >= 6:   score += 3
        elif k >= 5.5: score += 1
        elif k >= 3.5: score += 0
        elif k >= 3:   score += 1
        elif k >= 2.5: score += 2
        else:          score += 4

    # Creatinine
    if cr is not None:
        cr_pts = 0
        if   cr >= 3.5: cr_pts = 4
        elif cr >= 2.0: cr_pts = 3
        elif cr >= 1.5: cr_pts = 2
        elif cr >= 0.6: cr_pts = 0
        else:           cr_pts = 2
        if arf:
            cr_pts = min(cr_pts * 2, 4)
        score += cr_pts

    # Hematocrit
    if hct is not None:
        if   hct >= 60:  score += 4
        elif hct >= 50:  score += 2
        elif hct >= 46:  score += 1
        elif hct >= 30:  score += 0
        elif hct >= 20:  score += 2
        else:            score += 4

    # WBC (×10³/µL)
    if wbc is not None:
        if   wbc >= 40:  score += 4
        elif wbc >= 20:  score += 2
        elif wbc >= 15:  score += 1
        elif wbc >= 3:   score += 0
        elif wbc >= 1:   score += 2
        else:            score += 4

    # GCS (15 − GCS)
    score += max(0, 15 - int(gcs))

    # Age
    age_pts = (0 if age < 45 else 2 if age < 55 else 3 if age < 65 else 5 if age < 75 else 6)
    score += age_pts

    # Chronic health
    if chronic:
        st = surgery_type.lower() if surgery_type else ""
        score += 2 if "elective" in st else 5

    return _ok(score, "points")


# ─────────────────────────────────────────────────────────────────────────────
# ID 29 — PSI Score: Pneumonia Severity Index
# ─────────────────────────────────────────────────────────────────────────────

def calc_29(params: dict) -> dict:
    age = _get(params, "age")
    sex = _female(params)
    if None in (age, sex):
        return _missing("age, sex")

    score = age if not sex else (age - 10)

    def flag(*keys):
        return int(bool(_get(params, *keys, default=0, cast=int)))

    score += 10 * flag("nursing_home", "nursing home resident")
    score += 30 * flag("neoplastic_disease", "neoplastic disease")
    score += 20 * flag("liver_disease", "liver disease history",
                       "liver_disease_history")
    score += 10 * flag("chf", "congestive_heart_failure",
                        "congestive heart failure")
    score += 10 * flag("cerebrovascular_disease",
                       "cerebrovascular disease history")
    score += 10 * flag("renal_disease", "renal disease history",
                       "renal_disease_history")

    # Exam/vitals
    score += 20 * flag("altered_mental_status", "altered mental status")
    rr  = _get(params, "rr", "respiratory_rate", default=0)
    sbp = _get(params, "sbp", "systolic_blood_pressure",
               "systolic blood pressure", default=120)
    temp = _temp_c(_get(params, "temperature", "temp", default=37))
    hr   = _get(params, "hr", "heart_rate", "heart rate or pulse", default=80)
    score += 20 if rr >= 30 else 0
    score += 20 if sbp < 90 else 0
    score += 15 if temp < 35 or temp >= 40 else 0
    score += 10 if hr >= 125 else 0

    # Labs
    ph   = _get(params, "ph", default=7.4)
    bun  = _get(params, "bun", "blood_urea_nitrogen", default=0)
    na   = _get(params, "sodium", "serum_sodium", default=135)
    glu  = _get(params, "glucose", default=100)
    hct  = _get(params, "hematocrit", "hct", default=40)
    pao2 = _get(params, "pao2", default=80)

    score += 30 if ph < 7.35 else 0
    score += 20 if bun >= 30 else 0
    score += 20 if na < 130 else 0
    score += 10 if glu >= 250 else 0
    score += 10 if hct < 30 else 0
    score += 10 if pao2 < 60 else 0
    score += 10 * flag("pleural_effusion", "pleural effusion on x-ray")

    return _ok(score, "points")


# ─────────────────────────────────────────────────────────────────────────────
# ID 30 — Serum Osmolality
# ─────────────────────────────────────────────────────────────────────────────

def calc_30(params: dict) -> dict:
    na  = _get(params, "sodium", "serum_sodium", "na")
    glu = _glucose_mgdl(_get(params, "glucose", "serum_glucose"))
    bun = _get(params, "bun", "blood_urea_nitrogen",
               "blood urea nitrogen (bun)")
    if None in (na, glu, bun):
        return _missing("sodium, glucose, bun")
    osm = 2 * na + glu / 18 + bun / 2.8
    return _ok(round(osm, 4), "mOsm/kg")


# ─────────────────────────────────────────────────────────────────────────────
# ID 31 — HOMA-IR
# ─────────────────────────────────────────────────────────────────────────────

def calc_31(params: dict) -> dict:
    ins = _get(params, "insulin", "fasting_insulin", "serum_insulin")
    glu = _glucose_mgdl(_get(params, "glucose", "fasting_glucose", "serum_glucose"))
    if None in (ins, glu):
        return _missing("insulin, glucose")
    homa = (ins * glu) / 405
    return _ok(round(homa, 4), "")


# ─────────────────────────────────────────────────────────────────────────────
# ID 32 — Charlson Comorbidity Index (CCI)
# ─────────────────────────────────────────────────────────────────────────────

def calc_32(params: dict) -> dict:
    def flag(*keys):
        return int(bool(_get(params, *keys, default=0, cast=int)))

    age = _get(params, "age", default=0)
    age_pts = (0 if age < 50 else 1 if age < 60 else 2 if age < 70 else 3)

    score = (
        flag("mi", "myocardial_infarction")
        + flag("chf", "congestive_heart_failure")
        + flag("pvd", "peripheral_vascular_disease")
        + flag("cerebrovascular", "cerebrovascular_accident",
               "tia", "transient_ischemic_attacks_history")
        + flag("dementia")
        + flag("copd", "chronic_pulmonary_disease")
        + flag("connective_tissue_disease", "connective tissue disease")
        + flag("peptic_ulcer", "peptic ulcer disease")
        + flag("diabetes_mild", "diabetes mellitus")
        + flag("hemiplegia")
        + flag("ckd", "moderate_to_severe_chronic_kidney_disease",
               "moderate to severe chronic kidney disease")
        + 2 * flag("diabetes_complications")
        + 2 * flag("solid_tumor", "solid tumor")
        + 2 * flag("leukemia")
        + 2 * flag("lymphoma")
        + 3 * flag("liver_moderate", "liver disease severity")
        + 6 * flag("liver_severe")
        + 6 * flag("metastatic_solid_tumor")
        + 6 * flag("aids")
        + age_pts
    )
    return _ok(score, "points")


# ─────────────────────────────────────────────────────────────────────────────
# ID 33 — FeverPAIN Score for Strep Pharyngitis
# ─────────────────────────────────────────────────────────────────────────────

def calc_33(params: dict) -> dict:
    def flag(*keys):
        return int(bool(_get(params, *keys, default=0, cast=int)))

    fever  = flag("fever", "fever in past 24 hours")
    no_cough = flag("no_cough", "absence of cough",
                    "absence of cough or coryza",
                    "absence_of_cough_or_coryza")
    onset  = flag("onset_3_days", "symptom onset <=3 days",
                  "symptom_onset_3_days")
    purulent = flag("purulent_tonsils", "purulent tonsils")
    severe   = flag("severe_tonsil", "severe tonsil inflammation")

    return _ok(fever + no_cough + onset + purulent + severe, "points")


# ─────────────────────────────────────────────────────────────────────────────
# ID 36 — Caprini Score for Venous Thromboembolism (2005)
# ─────────────────────────────────────────────────────────────────────────────

def calc_36(params: dict) -> dict:
    def flag(*keys):
        return int(bool(_get(params, *keys, default=0, cast=int)))

    age  = _get(params, "age", default=0)
    bmi  = _get(params, "bmi", "body mass index (bmi)", default=0)

    # Age
    if   age >= 75: age_pts = 3
    elif age >= 61: age_pts = 2
    elif age >= 41: age_pts = 1
    else:           age_pts = 0

    # BMI >25
    bmi_pts = 1 if bmi > 25 else 0

    pts1 = (
        flag("minor_surgery")
        + flag("swollen_legs", "current swollen legs")
        + flag("varicose_veins", "varicose veins")
        + flag("pregnancy_postpartum")
        + flag("oral_contraceptives", "hormone_use")
        + flag("sepsis", "sepsis in the last month")
        + flag("copd", "chronic obstructive pulmonary disease")
        + flag("acute_mi", "acute myocardial infarction")
        + flag("chf", "congestive heart failure in the last month")
        + flag("ibd", "history of inflammatory bowel disease")
        + flag("bedridden_72h")
        + bmi_pts
    )

    pts2 = (
        flag("central_venous_access", "current central venous access")
        + flag("malignancy", "present or previous malignancy")
    )

    pts3 = (
        flag("vte_history", "previously documented pulmonary embolism",
             "previously documented deep vein thrombosis",
             "previous_vte")
        + flag("family_thrombosis", "family history of thrombosis")
        + flag("factor_v_leiden", "positive factor v leiden")
        + flag("prothrombin_20210a", "positive prothrombin 20210a")
        + flag("lupus_anticoagulant", "positive lupus anticoagulant")
        + flag("anticardiolipin", "elevated anticardiolipin antibody")
        + flag("homocysteine", "elevated serum homocysteine")
        + flag("hit", "heparin-induced thrombocytopenia")
        + flag("other_thrombophilia",
               "other congenital or acquired thrombophilia")
    )

    pts5 = (
        flag("stroke", "stroke in the last month")
        + flag("hip_fracture",
               "hip, pelvis, or leg fracture in the last month")
        + flag("multiple_trauma", "multiple trauma in the last month")
        + flag("spinal_injury",
               "acute spinal cord injury causing paralysis in the last month")
        + flag("plaster_cast",
               "immobilizing plaster cast in the last month")
        + flag("major_surgery", "major surgery in the last month")
    )

    sex  = _female(params)
    sex_pts = 1 if sex else 0  # female adds 1

    total = age_pts + pts1 + 2 * pts2 + 3 * pts3 + 5 * pts5 + sex_pts
    return _ok(total, "points")


# ─────────────────────────────────────────────────────────────────────────────
# ID 38 — Free Water Deficit
# ─────────────────────────────────────────────────────────────────────────────

def calc_38(params: dict) -> dict:
    na  = _get(params, "sodium", "serum_sodium", "na")
    wt  = _get(params, "weight", "weight_kg")
    age = _get(params, "age", default=40)
    sex = _female(params)
    if None in (na, wt, sex):
        return _missing("sodium, weight, sex")
    if sex:
        tbw_frac = 0.45 if age >= 60 else 0.5
    else:
        tbw_frac = 0.5  if age >= 60 else 0.6
    tbw = tbw_frac * wt
    fwd = tbw * (na / 140 - 1)
    return _ok(round(fwd, 4), "L")


# ─────────────────────────────────────────────────────────────────────────────
# ID 39 — Anion Gap
# ─────────────────────────────────────────────────────────────────────────────

def calc_39(params: dict) -> dict:
    na  = _get(params, "sodium", "na")
    cl  = _get(params, "chloride", "cl")
    hco3 = _get(params, "bicarbonate", "hco3",
                "hco3_serum", "serum bicarbonate")
    if None in (na, cl, hco3):
        return _missing("sodium, chloride, bicarbonate")
    ag = na - cl - hco3
    return _ok(round(ag, 4), "mEq/L")


# ─────────────────────────────────────────────────────────────────────────────
# ID 40 — Fractional Excretion of Sodium (FENa)
# ─────────────────────────────────────────────────────────────────────────────

def calc_40(params: dict) -> dict:
    na_s  = _get(params, "sodium", "serum_sodium", "na")
    na_u  = _get(params, "urine_sodium")
    cr_s  = _cr_mgdl(_get(params, "creatinine", "serum_creatinine", "scr"))
    # Urine creatinine is normally 40-300 mg/dL — do NOT apply µmol/L detection
    cr_u  = _get(params, "urine_creatinine")
    if None in (na_s, na_u, cr_s, cr_u):
        return _missing("sodium, urine_sodium, creatinine, urine_creatinine")
    fena = (na_u * cr_s) / (na_s * cr_u) * 100
    return _ok(round(fena, 4), "%")


# ─────────────────────────────────────────────────────────────────────────────
# ID 43 — SOFA Score
# ─────────────────────────────────────────────────────────────────────────────

def calc_43(params: dict) -> dict:
    # Respiratory: PaO2/FiO2
    pao2 = _get(params, "pao2")
    fio2 = _get(params, "fio2")
    on_vent = _get(params, "on_mechanical_ventilation",
                   "on mechanical ventilation", default=0, cast=int)
    on_cpap = _get(params, "cpap", "continuous positive airway pressure",
                   "continuous_positive_airway_pressure", default=0, cast=int)

    resp_pts = 0
    if pao2 is not None and fio2 is not None and fio2 > 0:
        pf = pao2 / fio2
        if   pf < 100 and (on_vent or on_cpap): resp_pts = 4
        elif pf < 200 and (on_vent or on_cpap): resp_pts = 3
        elif pf < 300: resp_pts = 2
        elif pf < 400: resp_pts = 1

    # Coagulation: platelets
    plt = _get(params, "platelets", "platelet_count",
               "platelet count", default=200)
    if   plt < 20:  plt_pts = 4
    elif plt < 50:  plt_pts = 3
    elif plt < 100: plt_pts = 2
    elif plt < 150: plt_pts = 1
    else:           plt_pts = 0

    # Liver: bilirubin
    bili = _get(params, "bilirubin", "total_bilirubin", default=0)
    if   bili >= 12: bili_pts = 4
    elif bili >= 6:  bili_pts = 3
    elif bili >= 2:  bili_pts = 2
    elif bili >= 1.2: bili_pts = 1
    else:            bili_pts = 0

    # Cardiovascular
    dopa = _get(params, "dopamine", "dopamine_dose",
                "dopamine dose", default=0)
    dobu = _get(params, "dobutamine", "dobutamine_dose",
                "dobutamine dose", default=0)
    epi  = _get(params, "epinephrine", "epinephrine_dose", default=0)
    norepi = _get(params, "norepinephrine", "norepinephrine_dose", default=0)
    map_  = _get(params, "map", "mean_arterial_pressure", default=70)
    hypot = _get(params, "hypotension", default=0, cast=int)

    if   (epi > 0.1 or norepi > 0.1):         cv_pts = 4
    elif (dopa > 15 or epi > 0 or norepi > 0): cv_pts = 3
    elif (dopa > 5 or dobu > 0):               cv_pts = 2
    elif (map_ < 70 or hypot):                 cv_pts = 1
    else:                                       cv_pts = 0

    # CNS: GCS
    gcs = _get(params, "gcs", "glasgow_coma_score",
               "glasgow coma score", default=15)
    if   gcs < 6:   cns_pts = 4
    elif gcs < 10:  cns_pts = 3
    elif gcs < 13:  cns_pts = 2
    elif gcs < 15:  cns_pts = 1
    else:           cns_pts = 0

    # Renal: creatinine / urine output
    cr  = _get(params, "creatinine", "serum_creatinine", default=0)
    uo  = _get(params, "urine_output", "urine_output_ml_day",
               "urine output", default=1000)
    if   cr >= 5.0 or uo < 200:   ren_pts = 4
    elif cr >= 3.5 or uo < 500:   ren_pts = 3
    elif cr >= 2.0:                ren_pts = 2
    elif cr >= 1.2:                ren_pts = 1
    else:                          ren_pts = 0

    total = resp_pts + plt_pts + bili_pts + cv_pts + cns_pts + ren_pts
    return _ok(total, "points")


# ─────────────────────────────────────────────────────────────────────────────
# ID 44 — LDL Calculated (Friedewald)
# ─────────────────────────────────────────────────────────────────────────────

def calc_44(params: dict) -> dict:
    tc  = _chol_mgdl(_get(params, "total_cholesterol", "total cholesterol"))
    hdl = _chol_mgdl(_get(params, "hdl", "high_density_lipoprotein_cholesterol",
                          "high-density lipoprotein cholesterol", "hdl_cholesterol"))
    tg  = _chol_mgdl(_get(params, "triglycerides", "tg"))
    if None in (tc, hdl, tg):
        return _missing("total_cholesterol, hdl, triglycerides")
    ldl = tc - hdl - (tg / 5)
    return _ok(round(ldl, 4), "mg/dL")


# ─────────────────────────────────────────────────────────────────────────────
# ID 45 — CURB-65
# ─────────────────────────────────────────────────────────────────────────────

def calc_45(params: dict) -> dict:
    age = _get(params, "age", default=0)
    bun = _get(params, "bun", "blood_urea_nitrogen",
               "blood urea nitrogen (bun)", default=0)
    rr  = _get(params, "rr", "respiratory_rate", default=0)
    sbp = _get(params, "sbp", "systolic_blood_pressure",
               "systolic blood pressure", default=120)
    dbp = _get(params, "dbp", "diastolic_blood_pressure",
               "diastolic blood pressure", default=80)
    confusion = _get(params, "confusion", default=0, cast=int)

    score = (
        int(bool(confusion))
        + (1 if bun > 19 else 0)
        + (1 if rr >= 30 else 0)
        + (1 if sbp < 90 or dbp <= 60 else 0)
        + (1 if age >= 65 else 0)
    )
    return _ok(score, "points")


# ─────────────────────────────────────────────────────────────────────────────
# ID 46 — Framingham Risk Score for Hard Coronary Heart Disease
# ─────────────────────────────────────────────────────────────────────────────

def calc_46(params: dict) -> dict:
    age  = _get(params, "age")
    tc   = _chol_mgdl(_get(params, "total_cholesterol", "total cholesterol"))
    hdl  = _chol_mgdl(_get(params, "hdl", "high_density_lipoprotein_cholesterol",
                           "high-density lipoprotein cholesterol", "hdl_cholesterol"))
    sbp  = _get(params, "sbp", "systolic_blood_pressure",
                "systolic blood pressure")
    bp_treated = _get(params, "bp_treated",
                      "blood pressure being treated with medicines",
                      "bp_treatment", default=0, cast=int)
    smoker = _get(params, "smoker", "smoking", "current_smoker",
                  default=0, cast=int)
    sex = _female(params)
    if None in (age, tc, hdl, sbp, sex):
        return _missing("age, total_cholesterol, hdl, sbp, sex")

    # ── Male tables ──────────────────────────────────────────────────────────
    if not sex:
        if   age < 35: age_pts = -9
        elif age < 40: age_pts = -4
        elif age < 45: age_pts = 0
        elif age < 50: age_pts = 3
        elif age < 55: age_pts = 6
        elif age < 60: age_pts = 8
        elif age < 65: age_pts = 10
        elif age < 70: age_pts = 11
        elif age < 75: age_pts = 12
        else:          age_pts = 13

        if age < 40:
            tc_pts = [0,4,7,9,11][min(4, [160,200,240,280,99999].index(
                next(x for x in [160,200,240,280,99999] if tc < x)))]
        elif age < 50:
            tc_pts = [0,3,5,6,8][min(4, [160,200,240,280,99999].index(
                next(x for x in [160,200,240,280,99999] if tc < x)))]
        elif age < 60:
            tc_pts = [0,2,3,4,5][min(4, [160,200,240,280,99999].index(
                next(x for x in [160,200,240,280,99999] if tc < x)))]
        elif age < 70:
            tc_pts = [0,1,1,2,3][min(4, [160,200,240,280,99999].index(
                next(x for x in [160,200,240,280,99999] if tc < x)))]
        else:
            tc_pts = [0,0,0,1,1][min(4, [160,200,240,280,99999].index(
                next(x for x in [160,200,240,280,99999] if tc < x)))]

        if   hdl >= 60: hdl_pts = -1
        elif hdl >= 50: hdl_pts = 0
        elif hdl >= 40: hdl_pts = 1
        else:           hdl_pts = 2

        if bp_treated:
            if   sbp >= 160: sbp_pts = 3
            elif sbp >= 140: sbp_pts = 2
            elif sbp >= 120: sbp_pts = 1
            else:            sbp_pts = 0
        else:
            if   sbp >= 160: sbp_pts = 2
            elif sbp >= 140: sbp_pts = 1
            elif sbp >= 130: sbp_pts = 1
            else:            sbp_pts = 0

        if age < 40:   smk_pts = 8
        elif age < 50: smk_pts = 5
        elif age < 60: smk_pts = 3
        else:          smk_pts = 1
        smk_pts = smk_pts if smoker else 0

        total = age_pts + tc_pts + hdl_pts + sbp_pts + smk_pts
        risk_table = {
            -99: 1, 0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2,
            7: 3, 8: 4, 9: 5, 10: 6, 11: 8, 12: 10, 13: 12,
            14: 16, 15: 20, 16: 25, 17: 30,
        }

    # ── Female tables ────────────────────────────────────────────────────────
    else:
        if   age < 35: age_pts = -7
        elif age < 40: age_pts = -3
        elif age < 45: age_pts = 0
        elif age < 50: age_pts = 3
        elif age < 55: age_pts = 6
        elif age < 60: age_pts = 8
        elif age < 65: age_pts = 10
        elif age < 70: age_pts = 12
        elif age < 75: age_pts = 14
        else:          age_pts = 16

        def _tc_pts_f(tc, age):
            idx = next((i for i, x in enumerate([160,200,240,280,99999]) if tc < x), 4)
            if age < 40:    row = [0,4,8,11,13]
            elif age < 50:  row = [0,3,6,8,10]
            elif age < 60:  row = [0,2,4,5,7]
            elif age < 70:  row = [0,1,2,3,4]
            else:           row = [0,1,1,2,2]
            return row[idx]
        tc_pts = _tc_pts_f(tc, age)

        if   hdl >= 60: hdl_pts = -1
        elif hdl >= 50: hdl_pts = 0
        elif hdl >= 40: hdl_pts = 1
        else:           hdl_pts = 2

        if bp_treated:
            if   sbp >= 160: sbp_pts = 6
            elif sbp >= 140: sbp_pts = 5
            elif sbp >= 130: sbp_pts = 4
            elif sbp >= 120: sbp_pts = 3
            else:            sbp_pts = 0
        else:
            if   sbp >= 160: sbp_pts = 4
            elif sbp >= 140: sbp_pts = 3
            elif sbp >= 130: sbp_pts = 2
            elif sbp >= 120: sbp_pts = 1
            else:            sbp_pts = 0

        if age < 40:   smk_pts = 9
        elif age < 50: smk_pts = 7
        elif age < 60: smk_pts = 4
        elif age < 70: smk_pts = 2
        else:          smk_pts = 1
        smk_pts = smk_pts if smoker else 0

        total = age_pts + tc_pts + hdl_pts + sbp_pts + smk_pts
        risk_table = {
            -99: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 2, 14: 2,
            15: 3, 16: 4, 17: 5, 18: 6, 19: 8, 20: 11, 21: 14,
            22: 17, 23: 22, 24: 27, 25: 30,
        }

    # Look up risk %
    keys = sorted(risk_table.keys())
    risk_pct = risk_table[keys[0]]
    for k in keys:
        if total >= k:
            risk_pct = risk_table[k]
    return _ok(risk_pct, "%", note=f"Total score: {total}")


# ─────────────────────────────────────────────────────────────────────────────
# ID 48 — PERC Rule for Pulmonary Embolism
# ─────────────────────────────────────────────────────────────────────────────

def calc_48(params: dict) -> dict:
    age  = _get(params, "age", default=0)
    hr   = _get(params, "hr", "heart_rate", "heart rate or pulse",
                "heart_rate_or_pulse", default=0)
    spo2 = _get(params, "spo2", "o2_saturation",
                "o2 saturation percentage", "oxygen_saturation", default=100)

    def flag(*keys):
        return int(bool(_get(params, *keys, default=0, cast=int)))

    count = (
        (1 if age >= 50 else 0)
        + (1 if hr >= 100 else 0)
        + (1 if spo2 < 95 else 0)
        + flag("unilateral_leg_swelling", "unilateral leg swelling")
        + flag("hemoptysis")
        + flag("recent_surgery", "recent surgery or trauma")
        + flag("previous_dvt_pe",
               "previously documented pulmonary embolism",
               "previously documented deep vein thrombosis")
        + flag("hormone_use", "hormone use")
    )
    return _ok(count, "criteria met")


# ─────────────────────────────────────────────────────────────────────────────
# ID 49 — Morphine Milligram Equivalents (MME)
# ─────────────────────────────────────────────────────────────────────────────

_MME_FACTORS = {
    "codeine": 0.15,
    "fentanyl_buccal": 0.13,
    "fentanyl_patch": 2.4,    # mcg/hr → MME/day
    "hydrocodone": 1.0,
    "hydromorphone": 4.0,
    "methadone": None,         # dose-dependent
    "morphine": 1.0,
    "oxycodone": 1.5,
    "oxymorphone": 3.0,
    "tapentadol": 0.4,
    "tramadol": 0.1,
}

def _methadone_factor(dose_day):
    if dose_day <= 20:   return 4
    elif dose_day <= 40: return 8
    elif dose_day <= 60: return 10
    else:                return 12

def calc_49(params: dict) -> dict:
    total_mme = 0.0
    for drug, factor in _MME_FACTORS.items():
        dose    = _get(params, f"{drug}_dose",
                       f"{drug.replace('_',' ')} dose", default=None)
        freq    = _get(params, f"{drug}_dose_per_day",
                       f"{drug.replace('_',' ')} dose per day", default=None)
        if dose is None and freq is None:
            continue
        # dose_per_day in dataset = frequency (times/day); total daily mg = dose × freq
        dose    = dose or 0.0
        freq    = freq or 1.0
        dose_day = dose * freq
        if dose_day == 0:
            continue
        if drug == "methadone":
            factor = _methadone_factor(dose_day)
        total_mme += dose_day * factor
    return _ok(round(total_mme, 4), "MME/day")


# ─────────────────────────────────────────────────────────────────────────────
# ID 51 — SIRS Criteria
# ─────────────────────────────────────────────────────────────────────────────

def calc_51(params: dict) -> dict:
    temp  = _temp_c(_get(params, "temperature", "temp", default=37))
    hr    = _get(params, "hr", "heart_rate", "heart rate or pulse",
                 "heart_rate_or_pulse", default=80)
    rr    = _get(params, "rr", "respiratory_rate", default=16)
    paco2 = _get(params, "paco2", default=40)
    wbc   = _plt_k(_get(params, "wbc", "white_blood_cell_count",
                        "white blood cell count", default=8))

    count = (
        (1 if temp > 38 or temp < 36 else 0)
        + (1 if hr > 90 else 0)
        + (1 if rr > 20 or paco2 < 32 else 0)
        + (1 if wbc > 12 or wbc < 4 else 0)
    )
    return _ok(count, "criteria met")


# ─────────────────────────────────────────────────────────────────────────────
# ID 56 — QTc Fridericia
# ─────────────────────────────────────────────────────────────────────────────

def calc_56(params: dict) -> dict:
    qt = _get(params, "qt_interval", "qt", "qt interval")
    hr = _get(params, "heart_rate", "hr", "heart rate or pulse",
              "heart_rate_or_pulse")
    if None in (qt, hr):
        return _missing("qt_interval, heart_rate")
    rr  = 60 / hr
    qtc = qt / (rr ** (1/3))
    return _ok(round(qtc, 4), "ms")


# ─────────────────────────────────────────────────────────────────────────────
# ID 57 — QTc Framingham
# ─────────────────────────────────────────────────────────────────────────────

def calc_57(params: dict) -> dict:
    qt = _get(params, "qt_interval", "qt", "qt interval")
    hr = _get(params, "heart_rate", "hr", "heart rate or pulse",
              "heart_rate_or_pulse")
    if None in (qt, hr):
        return _missing("qt_interval, heart_rate")
    rr  = 60 / hr  # seconds
    qtc = qt + 154 * (1 - rr)   # QT in ms → QTc in ms
    return _ok(round(qtc, 4), "ms")


# ─────────────────────────────────────────────────────────────────────────────
# ID 58 — QTc Hodges
# ─────────────────────────────────────────────────────────────────────────────

def calc_58(params: dict) -> dict:
    qt = _get(params, "qt_interval", "qt", "qt interval")
    hr = _get(params, "heart_rate", "hr", "heart rate or pulse",
              "heart_rate_or_pulse")
    if None in (qt, hr):
        return _missing("qt_interval, heart_rate")
    qtc = qt + 1.75 * (hr - 60)
    return _ok(round(qtc, 4), "ms")


# ─────────────────────────────────────────────────────────────────────────────
# ID 59 — QTc Rautaharju
# ─────────────────────────────────────────────────────────────────────────────

def calc_59(params: dict) -> dict:
    qt = _get(params, "qt_interval", "qt", "qt interval")
    hr = _get(params, "heart_rate", "hr", "heart rate or pulse",
              "heart_rate_or_pulse")
    if None in (qt, hr):
        return _missing("qt_interval, heart_rate")
    # QTcR = QT × (120 + HR) / 180
    qtc = qt * (120 + hr) / 180
    return _ok(round(qtc, 4), "ms")


# ─────────────────────────────────────────────────────────────────────────────
# ID 60 — Body Surface Area (Mosteller)
# ─────────────────────────────────────────────────────────────────────────────

def calc_60(params: dict) -> dict:
    wt = _get(params, "weight", "weight_kg")
    ht = _height_cm(_get(params, "height", "height_cm"))
    if None in (wt, ht):
        return _missing("weight, height")
    bsa = math.sqrt(ht * wt / 3600)
    return _ok(round(bsa, 4), "m²")


# ─────────────────────────────────────────────────────────────────────────────
# ID 61 — Target Weight
# ─────────────────────────────────────────────────────────────────────────────

def calc_61(params: dict) -> dict:
    bmi = _get(params, "bmi", "target_bmi", "body mass index (bmi)")
    ht  = _height_cm(_get(params, "height", "height_cm"))
    if None in (bmi, ht):
        return _missing("bmi, height")
    tw = bmi * (ht / 100) ** 2
    return _ok(round(tw, 4), "kg")


# ─────────────────────────────────────────────────────────────────────────────
# ID 62 — Adjusted Body Weight
# ─────────────────────────────────────────────────────────────────────────────

def calc_62(params: dict) -> dict:
    wt  = _get(params, "weight", "weight_kg", "actual_body_weight")
    ht  = _height_cm(_get(params, "height", "height_cm"))
    sex = _female(params)
    if None in (wt, ht, sex):
        return _missing("weight, height, sex")
    ht_in = ht / 2.54
    ibw = (45.5 if sex else 50.0) + 2.3 * (ht_in - 60)
    if wt <= ibw:
        return _ok(round(wt, 4), "kg", note="weight ≤ IBW; use actual weight")
    adj = ibw + 0.4 * (wt - ibw)
    return _ok(round(adj, 4), "kg")


# ─────────────────────────────────────────────────────────────────────────────
# ID 63 — Delta Gap
# ─────────────────────────────────────────────────────────────────────────────

def calc_63(params: dict) -> dict:
    """Delta Gap = Anion Gap − 12  (i.e. the excess anion gap above normal)."""
    na   = _get(params, "sodium", "na")
    cl   = _get(params, "chloride", "cl")
    hco3 = _get(params, "bicarbonate", "hco3")
    if None in (na, cl, hco3):
        return _missing("sodium, chloride, bicarbonate")
    ag = na - cl - hco3
    return _ok(round(ag - 12, 4), "mEq/L")


# ─────────────────────────────────────────────────────────────────────────────
# ID 64 — Delta Ratio
# ─────────────────────────────────────────────────────────────────────────────

def calc_64(params: dict) -> dict:
    na   = _get(params, "sodium", "na")
    cl   = _get(params, "chloride", "cl")
    hco3 = _get(params, "bicarbonate", "hco3")
    if None in (na, cl, hco3):
        return _missing("sodium, chloride, bicarbonate")
    ag = na - cl - hco3
    denom = 24 - hco3
    if denom == 0:
        return {"value": None, "unit": "", "note": "bicarbonate = 24; delta ratio undefined"}
    ratio = (ag - 12) / denom
    return _ok(round(ratio, 4), "")


# ─────────────────────────────────────────────────────────────────────────────
# ID 65 — Albumin Corrected Anion Gap
# ─────────────────────────────────────────────────────────────────────────────

def calc_65(params: dict) -> dict:
    na   = _get(params, "sodium", "na")
    cl   = _get(params, "chloride", "cl")
    hco3 = _get(params, "bicarbonate", "hco3")
    alb  = _albumin_gdl(_get(params, "albumin", "alb"))
    if None in (na, cl, hco3, alb):
        return _missing("sodium, chloride, bicarbonate, albumin")
    ag = na - cl - hco3
    corrected_ag = ag + 2.5 * (4.0 - alb)
    return _ok(round(corrected_ag, 4), "mEq/L")


# ─────────────────────────────────────────────────────────────────────────────
# ID 66 — Albumin Corrected Delta Gap
# ─────────────────────────────────────────────────────────────────────────────

def calc_66(params: dict) -> dict:
    """Albumin Corrected Delta Gap = Albumin Corrected AG − 12."""
    na   = _get(params, "sodium", "na")
    cl   = _get(params, "chloride", "cl")
    hco3 = _get(params, "bicarbonate", "hco3")
    alb  = _albumin_gdl(_get(params, "albumin", "alb"))
    if None in (na, cl, hco3, alb):
        return _missing("sodium, chloride, bicarbonate, albumin")
    ag = na - cl - hco3
    corr_ag = ag + 2.5 * (4.0 - alb)
    return _ok(round(corr_ag - 12, 4), "mEq/L")


# ─────────────────────────────────────────────────────────────────────────────
# ID 67 — Albumin Corrected Delta Ratio
# ─────────────────────────────────────────────────────────────────────────────

def calc_67(params: dict) -> dict:
    na   = _get(params, "sodium", "na")
    cl   = _get(params, "chloride", "cl")
    hco3 = _get(params, "bicarbonate", "hco3")
    alb  = _albumin_gdl(_get(params, "albumin", "alb"))
    if None in (na, cl, hco3, alb):
        return _missing("sodium, chloride, bicarbonate, albumin")
    ag = na - cl - hco3
    corr_ag = ag + 2.5 * (4.0 - alb)
    denom = 24 - hco3
    if denom == 0:
        return {"value": None, "unit": "", "note": "bicarbonate = 24; ratio undefined"}
    ratio = (corr_ag - 12) / denom
    return _ok(round(ratio, 4), "")


# ─────────────────────────────────────────────────────────────────────────────
# ID 68 — Estimated Date of Conception
# ─────────────────────────────────────────────────────────────────────────────

def calc_68(params: dict) -> dict:
    lmp_raw = _get(params, "last_menstrual_date", "lmp",
                   "last_menstrual_period", cast=str)
    if lmp_raw is None:
        return _missing("last_menstrual_date")
    try:
        lmp = date.fromisoformat(lmp_raw.strip())
    except ValueError:
        return {"value": None, "unit": "", "note": f"cannot parse date: {lmp_raw}"}
    edc = lmp + timedelta(days=14)
    return _ok(edc.isoformat(), "date")


# ─────────────────────────────────────────────────────────────────────────────
# ID 69 — Estimated Gestational Age
# ─────────────────────────────────────────────────────────────────────────────

def calc_69(params: dict) -> dict:
    lmp_raw     = _get(params, "last_menstrual_date", "lmp",
                       "last_menstrual_period", cast=str)
    current_raw = _get(params, "current_date", "today", "date", cast=str)
    if lmp_raw is None:
        return _missing("last_menstrual_date")
    if current_raw is None:
        return _missing("current_date")
    try:
        lmp     = date.fromisoformat(lmp_raw.strip())
        current = date.fromisoformat(current_raw.strip())
    except ValueError as e:
        return {"value": None, "unit": "", "note": str(e)}
    days  = (current - lmp).days
    weeks = days // 7
    rem   = days % 7
    return _ok(f"{weeks}w{rem}d", "weeks+days",
               note=f"{days} days total")


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY = {
    2: calc_2,   3: calc_3,   4: calc_4,   5: calc_5,
    6: calc_6,   7: calc_7,   8: calc_8,   9: calc_9,
    10: calc_10, 11: calc_11, 13: calc_13, 15: calc_15,
    16: calc_16, 17: calc_17, 18: calc_18, 19: calc_19,
    20: calc_20, 21: calc_21, 22: calc_22, 23: calc_23,
    24: calc_24, 25: calc_25, 26: calc_26, 27: calc_27,
    28: calc_28, 29: calc_29, 30: calc_30, 31: calc_31,
    32: calc_32, 33: calc_33, 36: calc_36, 38: calc_38,
    39: calc_39, 40: calc_40, 43: calc_43, 44: calc_44,
    45: calc_45, 46: calc_46, 48: calc_48, 49: calc_49,
    51: calc_51, 56: calc_56, 57: calc_57, 58: calc_58,
    59: calc_59, 60: calc_60, 61: calc_61, 62: calc_62,
    63: calc_63, 64: calc_64, 65: calc_65, 66: calc_66,
    67: calc_67, 68: calc_68, 69: calc_69,
}


def run_calculator(calculator_id: int, params: dict) -> dict:
    """
    Run the calculator for the given ID with the extracted params.
    Returns {"value": ..., "unit": ..., "note": ...}
    """
    fn = _REGISTRY.get(int(calculator_id))
    if fn is None:
        return {"value": None, "unit": "",
                "note": f"no calculator implemented for ID {calculator_id}"}
    try:
        return fn(params)
    except Exception as e:
        return {"value": None, "unit": "", "note": f"runtime error: {e}"}
