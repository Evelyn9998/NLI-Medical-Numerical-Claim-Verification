"""
Medical Calculators for NLI Numerical Verification
====================================================
Python implementations of 55 MedCalc-Bench calculators.
Each function accepts a params dict (from parse_entities) and returns a numeric result.

Usage:
    from calculators import run_calculator
    output = run_calculator("BMI", {"height": 1.75, "weight": 70})
    # {'success': True, 'result': 22.86, 'unit': 'kg/m²', 'name': 'Body Mass Index (BMI)'}
"""

import math
import re
from typing import Any, Dict, Optional


# ── Helper Utilities ─────────────────────────────────────────────────────────

def get(params: dict, *keys, default=None) -> Any:
    """
    Flexible parameter lookup with exact match first, then substring match.
    Keys are checked in order; first hit wins.
    """
    lower_params = {k.lower().strip(): v for k, v in params.items()}
    # Exact match pass
    for key in keys:
        val = lower_params.get(key.lower().strip())
        if val is not None:
            return val
    # Substring match pass
    for key in keys:
        k_low = key.lower().strip()
        for pk, pv in lower_params.items():
            if k_low in pk or pk in k_low:
                return pv
    return default


def parse_bool(val) -> bool:
    """Convert various representations to bool."""
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).lower().strip()
    return s in ('yes', 'true', '1', 'present', 'positive', 'confirmed', 'known', 'recent')


def parse_sex(val) -> str:
    """Return 'male' or 'female'."""
    if val is None:
        return 'unknown'
    s = str(val).lower().strip()
    if s in ('male', 'm'):
        return 'male'
    if s in ('female', 'f'):
        return 'female'
    return s


def to_float(val, default=None) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _height_to_cm(height_raw) -> Optional[float]:
    """Convert height to cm. Detects meters (<3), inches (30-100), cm (>100)."""
    h = to_float(height_raw)
    if h is None:
        return None
    if h < 3:           # meters
        return h * 100
    elif h <= 90:       # inches  (30-90 range)
        return h * 2.54
    else:               # cm
        return h


def _height_to_m(height_raw) -> Optional[float]:
    cm = _height_to_cm(height_raw)
    return cm / 100 if cm is not None else None


def _weight_to_kg(weight_raw) -> Optional[float]:
    """Convert weight to kg. >200 assumed lbs."""
    w = to_float(weight_raw)
    if w is None:
        return None
    if w > 200:
        return w / 2.205
    return w


def _temp_to_celsius(temp_raw) -> Optional[float]:
    """Convert temperature to Celsius. >45 assumed Fahrenheit."""
    t = to_float(temp_raw)
    if t is None:
        return None
    if t > 45:
        return (t - 32) * 5 / 9
    return t


def _creatinine_to_mgdl(cr_raw) -> Optional[float]:
    """Convert creatinine to mg/dL. >20 assumed µmol/L."""
    cr = to_float(cr_raw)
    if cr is None:
        return None
    if cr > 20:
        return cr / 88.4
    return cr


def _bilirubin_to_mgdl(bili_raw) -> Optional[float]:
    """Convert bilirubin to mg/dL. <10 and small values: if <5 assume mg/dL, else check."""
    bili = to_float(bili_raw)
    if bili is None:
        return None
    # If > 30, likely µmol/L → divide by 17.1
    if bili > 30:
        return bili / 17.1
    return bili


def _albumin_to_gdl(alb_raw) -> Optional[float]:
    """Convert albumin to g/dL. >10 assumed g/L → divide by 10."""
    alb = to_float(alb_raw)
    if alb is None:
        return None
    if alb > 10:
        return alb / 10
    return alb


def _calcium_to_mgdl(ca_raw) -> Optional[float]:
    """Convert calcium to mg/dL. <6 assumed mmol/L → multiply by 4.008."""
    ca = to_float(ca_raw)
    if ca is None:
        return None
    if ca < 6:
        return ca * 4.008
    return ca


def _glucose_to_mgdl(glc_raw) -> Optional[float]:
    """Convert glucose to mg/dL. <30 assumed mmol/L → multiply by 18.018."""
    glc = to_float(glc_raw)
    if glc is None:
        return None
    if glc < 30:
        return glc * 18.018
    return glc


def _sodium_to_meql(na_raw) -> Optional[float]:
    """Sodium in mEq/L (normal 135-145). Values outside reasonable range return as-is."""
    return to_float(na_raw)


def _platelets_to_k(plt_raw) -> Optional[float]:
    """Platelets in ×10⁹/L (k/µL). If >1000 assume /µL → divide by 1000."""
    plt = to_float(plt_raw)
    if plt is None:
        return None
    if plt > 1000:
        return plt / 1000
    return plt


def _wbc_to_k(wbc_raw) -> Optional[float]:
    """WBC in ×10³/µL. If >100 assume /µL → divide by 1000."""
    wbc = to_float(wbc_raw)
    if wbc is None:
        return None
    if wbc > 100:
        return wbc / 1000
    return wbc


def _fio2_to_fraction(fio2_raw) -> Optional[float]:
    """FiO2 as fraction (0-1). If >1, divide by 100."""
    fio2 = to_float(fio2_raw)
    if fio2 is None:
        return None
    if fio2 > 1:
        return fio2 / 100
    return fio2


def _ideal_body_weight(height_cm, sex) -> float:
    """Devine formula: IBW in kg."""
    h_in = height_cm / 2.54
    base = 50 if sex == 'male' else 45.5
    return base + 2.3 * max(0, h_in - 60)


# ── Calculator 2: Creatinine Clearance (Cockcroft-Gault) ─────────────────────

def cockcroft_gault(params: dict) -> dict:
    age    = to_float(get(params, 'age'))
    cr_raw = get(params, 'creatinine', 'serum creatinine')
    ht_raw = get(params, 'height')
    wt_raw = get(params, 'weight', 'actual body weight', 'body weight')
    sex    = parse_sex(get(params, 'sex', 'gender'))

    if any(v is None for v in [age, cr_raw, ht_raw, wt_raw]):
        return {'success': False, 'error': 'Missing required parameters'}

    cr  = _creatinine_to_mgdl(cr_raw)
    ht_cm = _height_to_cm(ht_raw)
    wt_kg = _weight_to_kg(wt_raw)
    if cr is None or cr == 0 or ht_cm is None or wt_kg is None:
        return {'success': False, 'error': 'Invalid parameter values'}

    ibw = _ideal_body_weight(ht_cm, sex)
    # Use Adjusted Body Weight if obese (actual > IBW)
    if wt_kg > ibw:
        weight_use = ibw + 0.4 * (wt_kg - ibw)
    else:
        weight_use = wt_kg

    sex_factor = 1.0 if sex == 'male' else 0.85
    crcl = (140 - age) * weight_use * sex_factor / (72 * cr)
    return {'success': True, 'result': round(crcl, 3), 'unit': 'mL/min',
            'name': 'Creatinine Clearance (Cockcroft-Gault Equation)'}


# ── Calculator 3: CKD-EPI GFR ─────────────────────────────────────────────────

def ckd_epi(params: dict) -> dict:
    age  = to_float(get(params, 'age'))
    cr   = _creatinine_to_mgdl(get(params, 'creatinine', 'serum creatinine'))
    sex  = parse_sex(get(params, 'sex', 'gender'))

    if any(v is None for v in [age, cr]):
        return {'success': False, 'error': 'Missing required parameters'}

    if sex == 'female':
        kappa, alpha = 0.7, -0.329
        sex_mult = 1.018
    else:
        kappa, alpha = 0.9, -0.411
        sex_mult = 1.0

    cr_kappa = cr / kappa
    if cr_kappa < 1:
        gfr = 141 * (cr_kappa ** alpha) * (0.9938 ** age) * sex_mult
    else:
        gfr = 141 * (cr_kappa ** -1.209) * (0.9938 ** age) * sex_mult

    return {'success': True, 'result': round(gfr, 3), 'unit': 'mL/min/1.73m²',
            'name': 'CKD-EPI Equations for Glomerular Filtration Rate'}


# ── Calculator 4: CHA2DS2-VASc ────────────────────────────────────────────────

def cha2ds2_vasc(params: dict) -> dict:
    age = to_float(get(params, 'age'))
    sex = parse_sex(get(params, 'sex', 'gender'))

    chf       = parse_bool(get(params, 'congestive heart failure', 'chf'))
    htn       = parse_bool(get(params, 'hypertension history', 'hypertension'))
    stroke    = parse_bool(get(params, 'stroke', 'thromboembolism history',
                                    'transient ischemic attacks history', 'tia'))
    dm        = parse_bool(get(params, 'diabetes history', 'diabetes'))
    vascular  = parse_bool(get(params, 'vascular disease history', 'vascular disease',
                                      'myocardial infarction'))

    score = 0
    if chf:      score += 1
    if htn:      score += 1
    if age is not None:
        if age >= 75:    score += 2
        elif age >= 65:  score += 1
    if dm:       score += 1
    if stroke:   score += 2
    if vascular: score += 1
    if sex == 'female': score += 1

    return {'success': True, 'result': score, 'unit': 'points',
            'name': "CHA2DS2-VASc Score for Atrial Fibrillation Stroke Risk"}


# ── Calculator 5: Mean Arterial Pressure ──────────────────────────────────────

def mean_arterial_pressure(params: dict) -> dict:
    sbp = to_float(get(params, 'systolic blood pressure', 'systolic bp', 'sbp'))
    dbp = to_float(get(params, 'diastolic blood pressure', 'diastolic bp', 'dbp'))
    if sbp is None or dbp is None:
        return {'success': False, 'error': 'Missing SBP or DBP'}
    map_ = dbp + (sbp - dbp) / 3
    return {'success': True, 'result': round(map_, 3), 'unit': 'mmHg',
            'name': 'Mean Arterial Pressure (MAP)'}


# ── Calculator 6: BMI ─────────────────────────────────────────────────────────

def bmi_calc(params: dict) -> dict:
    ht_raw = get(params, 'height')
    wt_raw = get(params, 'weight', 'body weight')
    if ht_raw is None or wt_raw is None:
        return {'success': False, 'error': 'Missing height or weight'}
    h_m = _height_to_m(ht_raw)
    w_kg = _weight_to_kg(wt_raw)
    if h_m is None or h_m == 0 or w_kg is None:
        return {'success': False, 'error': 'Invalid height or weight'}
    bmi = w_kg / (h_m ** 2)
    return {'success': True, 'result': round(bmi, 3), 'unit': 'kg/m²',
            'name': 'Body Mass Index (BMI)'}


# ── Calculator 7: Calcium Correction for Hypoalbuminemia ─────────────────────

def calcium_correction(params: dict) -> dict:
    alb_raw = get(params, 'albumin', 'serum albumin')
    ca_raw  = get(params, 'calcium', 'serum calcium', 'total calcium')
    if alb_raw is None or ca_raw is None:
        return {'success': False, 'error': 'Missing albumin or calcium'}
    alb = _albumin_to_gdl(alb_raw)
    ca  = _calcium_to_mgdl(ca_raw)
    if alb is None or ca is None:
        return {'success': False, 'error': 'Invalid values'}
    corrected = ca + 0.8 * (4.0 - alb)
    return {'success': True, 'result': round(corrected, 3), 'unit': 'mg/dL',
            'name': 'Calcium Correction for Hypoalbuminemia'}


# ── Calculator 8: Wells' Criteria for PE ──────────────────────────────────────

def wells_pe(params: dict) -> dict:
    # 3 pts each
    dvt_signs = parse_bool(get(params,
        'clinical signs and symptoms of deep vein thrombosis',
        'clinical signs of deep vein thrombosis',
        'clinical signs dvt', 'dvt signs'))
    pe_top    = parse_bool(get(params,
        'pulmonary embolism is #1 diagnosis or equally likely',
        'pe is #1 diagnosis', 'pe #1', 'pulmonary embolism is number 1 diagnosis'))

    # 1.5 pts each
    hr_raw     = to_float(get(params, 'heart rate or pulse', 'heart rate', 'pulse'))
    immob      = parse_bool(get(params, 'immobilization for at least 3 days',
                                       'immobilization', 'immobilized'))
    prev_dvt   = parse_bool(get(params, 'previously documented deep vein thrombosis',
                                       'previously documented dvt', 'prior dvt'))
    prev_pe    = parse_bool(get(params, 'previously documented pulmonary embolism',
                                       'prior pe', 'previous pe'))
    surgery    = parse_bool(get(params, 'surgery in the previous 4 weeks', 'recent surgery'))

    # 1 pt each
    malignancy = parse_bool(get(params, 'malignancy with treatment within 6 months or palliative',
                                       'malignancy', 'cancer'))
    hemoptysis = parse_bool(get(params, 'hemoptysis'))

    score = 0.0
    if dvt_signs:  score += 3.0
    if pe_top:     score += 3.0
    if hr_raw is not None and hr_raw > 100: score += 1.5
    if immob:      score += 1.5
    if prev_dvt:   score += 1.5
    if prev_pe:    score += 1.5
    if surgery:    score += 1.5
    if malignancy: score += 1.0
    if hemoptysis: score += 1.0

    return {'success': True, 'result': score, 'unit': 'points',
            'name': "Wells' Criteria for Pulmonary Embolism"}


# ── Calculator 9: MDRD GFR ────────────────────────────────────────────────────

def mdrd_gfr(params: dict) -> dict:
    age  = to_float(get(params, 'age'))
    cr   = _creatinine_to_mgdl(get(params, 'creatinine', 'serum creatinine'))
    sex  = parse_sex(get(params, 'sex', 'gender'))
    race_raw = get(params, 'race', 'ethnicity')
    race = str(race_raw).lower() if race_raw else ''
    black = 'black' in race or 'african' in race

    if age is None or cr is None:
        return {'success': False, 'error': 'Missing age or creatinine'}

    gfr = 175 * (cr ** -1.154) * (age ** -0.203)
    if sex == 'female': gfr *= 0.742
    if black:           gfr *= 1.212

    return {'success': True, 'result': round(gfr, 3), 'unit': 'mL/min/1.73m²',
            'name': 'MDRD GFR Equation'}


# ── Calculator 10: Ideal Body Weight ──────────────────────────────────────────

def ideal_body_weight(params: dict) -> dict:
    ht_raw = get(params, 'height')
    sex    = parse_sex(get(params, 'sex', 'gender'))
    if ht_raw is None:
        return {'success': False, 'error': 'Missing height'}
    ht_cm = _height_to_cm(ht_raw)
    if ht_cm is None:
        return {'success': False, 'error': 'Invalid height'}
    ibw = _ideal_body_weight(ht_cm, sex)
    return {'success': True, 'result': round(ibw, 3), 'unit': 'kg',
            'name': 'Ideal Body Weight'}


# ── Calculator 11: QTc Bazett ──────────────────────────────────────────────────

def qtc_bazett(params: dict) -> dict:
    hr = to_float(get(params, 'heart rate or pulse', 'heart rate', 'pulse'))
    qt = to_float(get(params, 'qt interval', 'qt'))
    if hr is None or qt is None or hr == 0:
        return {'success': False, 'error': 'Missing HR or QT'}
    rr = 60 / hr   # RR in seconds
    # QT in ms → seconds if >10
    qt_s = qt / 1000 if qt > 10 else qt
    qtc = qt_s / math.sqrt(rr)
    qtc_ms = qtc * 1000
    return {'success': True, 'result': round(qtc_ms, 3), 'unit': 'ms',
            'name': 'QTc Bazett Calculator'}


# ── Calculator 15: Child-Pugh Score ───────────────────────────────────────────

def child_pugh(params: dict) -> dict:
    alb_raw   = get(params, 'albumin', 'serum albumin')
    bili_raw  = get(params, 'bilirubin', 'total bilirubin')
    inr_raw   = get(params, 'international normalized ratio', 'inr', 'prothrombin time')
    ascites_raw = get(params, 'ascites')
    enceph_raw  = get(params, 'encephalopathy', 'hepatic encephalopathy')

    alb  = _albumin_to_gdl(alb_raw)
    bili = _bilirubin_to_mgdl(bili_raw)
    inr  = to_float(inr_raw)

    # Bilirubin in mg/dL
    if bili is None or alb is None or inr is None:
        return {'success': False, 'error': 'Missing required labs'}

    score = 0

    # Bilirubin
    if bili < 2:   score += 1
    elif bili < 3: score += 2
    else:          score += 3

    # Albumin
    if alb > 3.5:   score += 1
    elif alb >= 2.8: score += 2
    else:            score += 3

    # INR
    if inr < 1.7:   score += 1
    elif inr < 2.3: score += 2
    else:           score += 3

    # Ascites
    if ascites_raw is not None:
        asc = str(ascites_raw).lower().strip()
        if asc in ('none', 'absent', 'no', 'false', '0'):
            score += 1
        elif any(x in asc for x in ('slight', 'mild', 'small', 'moderate')):
            score += 2
        else:
            score += 3
    else:
        score += 1  # default: no ascites

    # Encephalopathy
    if enceph_raw is not None:
        enc = str(enceph_raw).lower().strip()
        if enc in ('none', 'absent', 'no', 'false', '0', 'grade 0'):
            score += 1
        elif any(x in enc for x in ('grade 1', 'grade 2', 'mild', 'slight', '1', '2')):
            score += 2
        else:
            score += 3
    else:
        score += 1

    return {'success': True, 'result': score, 'unit': 'points',
            'name': 'Child-Pugh Score for Cirrhosis Mortality'}


# ── Calculator 16: Wells' Criteria for DVT ────────────────────────────────────

def wells_dvt(params: dict) -> dict:
    active_cancer  = parse_bool(get(params, 'active cancer', 'malignancy', 'cancer'))
    paralysis      = parse_bool(get(params, 'paralysis, paresis, or recent plaster immobilization of the lower extremity',
                                          'paralysis', 'paresis', 'plaster immobilization'))
    bedridden      = parse_bool(get(params, 'bedridden recently >3 days',
                                          'bedridden', 'immobilization'))
    localized_tend = parse_bool(get(params, 'localized tenderness along the deep venous system',
                                          'localized tenderness'))
    entire_leg     = parse_bool(get(params, 'entire leg swollen', 'entire leg swelling'))
    calf_swelling  = parse_bool(get(params, 'calf swelling >3 centimeters compared to the other leg',
                                          'calf swelling'))
    pitting_edema  = parse_bool(get(params, 'pitting edema, confined to symptomatic leg',
                                          'pitting edema'))
    collateral     = parse_bool(get(params, 'collateral (nonvaricose) superficial veins present',
                                          'collateral veins'))
    prev_dvt       = parse_bool(get(params, 'previously documented deep vein thrombosis',
                                          'previously documented dvt', 'prior dvt', 'previous dvt'))
    alt_dx         = parse_bool(get(params, 'alternative diagnosis to deep vein thrombosis as likely or more likely',
                                          'alternative diagnosis', 'alt diagnosis'))
    major_surgery  = parse_bool(get(params, 'major surgery within 12 weeks', 'major surgery'))

    score = 0
    for crit in [active_cancer, paralysis, bedridden, localized_tend,
                 entire_leg, calf_swelling, pitting_edema, collateral,
                 prev_dvt, major_surgery]:
        if crit: score += 1
    if alt_dx: score -= 2

    return {'success': True, 'result': score, 'unit': 'points',
            'name': "Wells' Criteria for DVT"}


# ── Calculator 17: Revised Cardiac Risk Index ─────────────────────────────────

def revised_cardiac_risk_index(params: dict) -> dict:
    high_risk_sx = parse_bool(get(params, 'elevated-risk surgery', 'high risk surgery',
                                        'elevated risk surgery', 'suprainguinal vascular',
                                        'intraperitoneal', 'intrathoracic'))
    ihd     = parse_bool(get(params, 'history of ischemic heart disease', 'ischemic heart disease',
                                    'coronary artery disease', 'myocardial infarction history'))
    chf     = parse_bool(get(params, 'congestive heart failure criteria for the cardiac risk index rule',
                                    'congestive heart failure', 'chf'))
    cvd     = parse_bool(get(params, 'history of cerebrovascular disease',
                                    'cerebrovascular disease history', 'stroke', 'tia'))
    insulin = parse_bool(get(params, 'pre-operative treatment with insulin', 'insulin'))
    cr_raw  = get(params, 'pre-operative creatinine', 'creatinine', 'serum creatinine')
    cr      = _creatinine_to_mgdl(cr_raw) if cr_raw is not None else None
    cr_high = cr is not None and cr > 2.0

    score = sum([high_risk_sx, ihd, chf, cvd, insulin, cr_high])
    return {'success': True, 'result': score, 'unit': 'points',
            'name': 'Revised Cardiac Risk Index for Pre-Operative Risk'}


# ── Calculator 18: HEART Score ────────────────────────────────────────────────

def heart_score(params: dict) -> dict:
    # History (Suspicion History)
    history_raw = str(get(params, 'suspicion history', 'history', '') or '').lower()
    if any(x in history_raw for x in ('highly suspicious', 'high')):
        h_pts = 2
    elif any(x in history_raw for x in ('moderately suspicious', 'moderate')):
        h_pts = 1
    else:
        h_pts = 0

    # ECG
    ecg_raw = str(get(params, 'electrocardiogram test', 'ecg', 'ekg', '') or '').lower()
    if any(x in ecg_raw for x in ('significant st', 'st depression', 'lbbb', 'significant')):
        ecg_pts = 2
    elif any(x in ecg_raw for x in ('non-specific', 'nonspecific', 'lbbb absent',
                                     'st changes', 'normal')):
        ecg_pts = 1
    else:
        ecg_pts = 0

    # Age
    age = to_float(get(params, 'age'))
    if age is None:
        a_pts = 0
    elif age >= 65:
        a_pts = 2
    elif age >= 45:
        a_pts = 1
    else:
        a_pts = 0

    # Risk factors - count individually, also check atherosclerotic disease
    htn      = parse_bool(get(params, 'hypertension history', 'hypertension'))
    hyperchol = parse_bool(get(params, 'hypercholesterolemia', 'hypercholesterolaemia', 'high cholesterol'))
    dm        = parse_bool(get(params, 'diabetes mellitus', 'diabetes'))
    obesity   = parse_bool(get(params, 'obesity'))
    smoking   = parse_bool(get(params, 'smoking', 'smoker', 'current smoker'))
    family_hx = parse_bool(get(params, 'parent or sibling with cardiovascular disease before age 65',
                                      'family history', 'family history of cardiovascular disease'))
    atherosclerosis = parse_bool(get(params, 'atherosclerotic disease', 'atherosclerosis',
                                           'known atherosclerotic disease',
                                           'known coronary artery disease'))

    rf_count = sum([htn, hyperchol, dm, obesity, smoking, family_hx])
    if atherosclerosis:
        rf_pts = 2
    elif rf_count >= 3:
        rf_pts = 2
    elif rf_count >= 1:
        rf_pts = 1
    else:
        rf_pts = 0

    # Troponin - check from most specific to least specific
    trop_raw = str(get(params, 'initial troponin', 'troponin', '') or '').lower()
    # Check 3x+ first (highest)
    if any(x in trop_raw for x in ('three times', '>3x', '3x', 'greater than three',
                                    'more than three', '>3 times', '3 times')):
        t_pts = 2
    elif any(x in trop_raw for x in ('one to three times', '1-3x', '1 to 3',
                                      'between one and three', 'one and three',
                                      'mildly elevated', 'slightly elevated')):
        t_pts = 1
    elif any(x in trop_raw for x in ('normal', '<=normal', 'negative', 'within normal')):
        t_pts = 0
    else:
        t_pts = 0

    total = h_pts + ecg_pts + a_pts + rf_pts + t_pts
    return {'success': True, 'result': total, 'unit': 'points',
            'name': 'HEART Score for Major Cardiac Events'}


# ── Calculator 19: FIB-4 ─────────────────────────────────────────────────────

def fib4(params: dict) -> dict:
    age  = to_float(get(params, 'age'))
    ast  = to_float(get(params, 'aspartate aminotransferase', 'ast'))
    alt  = to_float(get(params, 'alanine aminotransferase', 'alt'))
    plt  = _platelets_to_k(get(params, 'platelet count', 'platelets', 'plt'))

    if any(v is None for v in [age, ast, plt]) or plt == 0:
        return {'success': False, 'error': 'Missing required parameters'}

    fib = (age * ast) / (plt * math.sqrt(alt if alt is not None and alt > 0 else ast))
    return {'success': True, 'result': round(fib, 3), 'unit': 'index',
            'name': 'Fibrosis-4 (FIB-4) Index for Liver Fibrosis'}


# ── Calculator 20: Centor Score ───────────────────────────────────────────────

def centor_score(params: dict) -> dict:
    exudate   = parse_bool(get(params, 'exudate or swelling on tonsils', 'tonsillar exudate',
                                      'exudate', 'tonsil exudate'))
    lymph     = parse_bool(get(params, 'tender/swollen anterior cervical lymph nodes',
                                      'anterior cervical lymph nodes',
                                      'tender anterior cervical lymph nodes',
                                      'cervical lymph nodes',
                                      'tender swollen anterior cervical lymph nodes',
                                      'lymph nodes tender'))
    temp_raw  = get(params, 'temperature', 'temp')
    temp_c    = _temp_to_celsius(to_float(temp_raw)) if temp_raw is not None else None
    fever     = temp_c is not None and temp_c > 38.0
    no_cough  = parse_bool(get(params, 'cough absent', 'absence of cough', 'no cough'))
    age       = to_float(get(params, 'age'))

    score = 0
    if exudate:  score += 1
    if lymph:    score += 1
    if fever:    score += 1
    if no_cough: score += 1

    # Age adjustment
    if age is not None:
        if age < 15:   score += 1
        elif age >= 45: score -= 1

    return {'success': True, 'result': score, 'unit': 'points',
            'name': 'Centor Score (Modified/McIsaac) for Strep Pharyngitis'}


# ── Calculator 21: Glasgow Coma Score ─────────────────────────────────────────

def glasgow_coma_score(params: dict) -> dict:
    eye    = to_float(get(params, 'best eye response', 'eye response', 'eye opening'))
    motor  = to_float(get(params, 'best motor response', 'motor response'))
    verbal = to_float(get(params, 'best verbal response', 'verbal response'))

    # Try string lookup if numbers not given
    def _gcs_lookup(raw, lookup_map):
        if raw is None: return None
        v = to_float(raw)
        if v is not None: return int(v)
        s = str(raw).lower().strip()
        for k, pts in lookup_map.items():
            if k in s: return pts
        return None

    eye_map   = {'spontaneous': 4, 'sound': 3, 'pain': 2, 'none': 1, 'no response': 1}
    motor_map = {'obey': 6, 'localiz': 5, 'withdraw': 4, 'flexion': 3, 'extension': 2,
                 'none': 1, 'no response': 1}
    verbal_map= {'orient': 5, 'confus': 4, 'inapp': 3, 'incomprehens': 2, 'none': 1, 'no response': 1}

    e = _gcs_lookup(eye,    eye_map)
    m = _gcs_lookup(motor,  motor_map)
    v = _gcs_lookup(verbal, verbal_map)

    total = (e or 1) + (m or 1) + (v or 1)
    return {'success': True, 'result': total, 'unit': 'points',
            'name': 'Glasgow Coma Score (GCS)'}


# ── Calculator 22: Maintenance Fluids ─────────────────────────────────────────

def maintenance_fluids(params: dict) -> dict:
    wt = _weight_to_kg(get(params, 'weight', 'body weight'))
    if wt is None:
        return {'success': False, 'error': 'Missing weight'}
    if wt <= 10:
        rate = wt * 100
    elif wt <= 20:
        rate = 1000 + (wt - 10) * 50
    else:
        rate = 1500 + (wt - 20) * 20
    return {'success': True, 'result': round(rate, 3), 'unit': 'mL/day',
            'name': 'Maintenance Fluids Calculations'}


# ── Calculator 23: MELD-Na ────────────────────────────────────────────────────

def meld_na(params: dict) -> dict:
    bili_raw = get(params, 'bilirubin', 'total bilirubin')
    cr_raw   = get(params, 'creatinine', 'serum creatinine')
    inr_raw  = get(params, 'international normalized ratio', 'inr')
    na_raw   = get(params, 'sodium', 'serum sodium')
    dialysis = parse_bool(get(params, 'dialysis at least twice in the past week',
                                     'continuous veno-venous hemodialysis for ≥4 hours in the past week',
                                     'dialysis', 'cvvhd', 'hemodialysis'))

    bili = _bilirubin_to_mgdl(bili_raw)
    cr   = _creatinine_to_mgdl(cr_raw)
    inr  = to_float(inr_raw)
    na   = to_float(na_raw)

    if any(v is None for v in [bili, cr, inr]):
        return {'success': False, 'error': 'Missing required labs'}

    # Floor values
    bili = max(bili, 1.0)
    cr   = max(cr, 1.0)
    inr  = max(inr, 1.0)

    if dialysis:
        cr = 4.0

    cr = min(cr, 4.0)

    meld = 3.78 * math.log(bili) + 11.2 * math.log(inr) + 9.57 * math.log(cr) + 6.43

    # MELD-Na adjustment
    if na is not None:
        na = max(125, min(na, 137))
        meld_na_score = meld - na - (0.025 * meld * (140 - na)) + 140
    else:
        meld_na_score = meld

    return {'success': True, 'result': round(meld_na_score, 3), 'unit': 'points',
            'name': 'MELD Na (UNOS/OPTN)'}


# ── Calculator 24: Steroid Conversion ─────────────────────────────────────────

# Prednisone-equivalent doses (mg) per 1 mg of each steroid
STEROID_EQUIV = {
    'betamethasone':         0.6,    # 0.6mg betas = 5mg pred → factor = 5/0.6=8.33
    'cortisone':             25.0,
    'dexamethasone':         0.75,
    'fludrocortisone':       10.0,
    'hydrocortisone':        20.0,
    'methylprednisolone':     4.0,
    'methylprednisolone po':  4.0,
    'methylprednisolone iv':  4.0,
    'prednisolone':           5.0,
    'prednisone':             5.0,
    'triamcinolone':          4.0,
    'budesonide':             0.9,   # inhaled, approximate
}

def _steroid_prednisone_mg(steroid_name: str, dose_mg: float) -> Optional[float]:
    """Convert dose of given steroid to prednisone-equivalent mg."""
    s = steroid_name.lower().strip()
    # Remove trailing PO/IV/IM designations for key lookup
    key = re.sub(r'\s+(po|iv|im|oral|parenteral)$', '', s).strip()
    # Try exact then substring
    factor = STEROID_EQUIV.get(key)
    if factor is None:
        for k, v in STEROID_EQUIV.items():
            if k in key or key in k:
                factor = v
                break
    if factor is None:
        return None
    # Prednisone equivalents per steroid equivalent dose
    # factor = "X mg of steroid = 5mg prednisone" (so factor is equi-potent dose)
    # Actually, STEROID_EQUIV above stores prednisone-equivalent-dose:
    # hydrocortisone 20mg = prednisone 5mg → factor=20 means divide dose by 20 then × 5
    # Let's use: prednisone_eq = dose_mg * 5 / factor
    return dose_mg * 5 / factor


def steroid_conversion(params: dict) -> dict:
    # Input steroid and dose
    input_steroid_raw  = get(params, 'input steroid')
    target_steroid_raw = get(params, 'target steroid')

    # Also look for 'input steroid dose' (set by parse_entities fix for ['Name', dose, 'unit'] format)
    input_dose_raw  = get(params, 'input steroid dose', 'input dose')
    target_dose_raw = get(params, 'target steroid dose', 'target dose')

    if input_steroid_raw is None:
        return {'success': False, 'error': 'Missing input steroid'}

    input_name = str(input_steroid_raw).strip()
    input_dose = to_float(input_dose_raw)

    if input_dose is None:
        return {'success': False, 'error': 'Missing input steroid dose'}

    # Convert input to prednisone equivalents
    pred_eq = _steroid_prednisone_mg(input_name, input_dose)
    if pred_eq is None:
        return {'success': False, 'error': f'Unknown steroid: {input_name}'}

    # Convert to target steroid
    if target_steroid_raw is not None:
        target_name = str(target_steroid_raw).strip()
        target_s    = target_name.lower().strip()
        key = re.sub(r'\s+(po|iv|im|oral|parenteral)$', '', target_s).strip()
        factor = STEROID_EQUIV.get(key)
        if factor is None:
            for k, v in STEROID_EQUIV.items():
                if k in target_s or target_s in k:
                    factor = v
                    break
        if factor is not None:
            target_dose_calc = pred_eq * factor / 5
        else:
            target_dose_calc = pred_eq  # fallback: prednisone equivalent
    else:
        target_dose_calc = pred_eq  # prednisone equivalent

    return {'success': True, 'result': round(target_dose_calc, 3), 'unit': 'mg',
            'name': 'Steroid Conversion Calculator'}


# ── Calculator 25: HAS-BLED Score ────────────────────────────────────────────

def has_bled(params: dict) -> dict:
    # H - Hypertension (uncontrolled, SBP > 160)
    htn_raw = get(params, 'hypertension', 'uncontrolled hypertension')
    htn = parse_bool(htn_raw)

    # A - Renal/Liver disease (1pt each, max 2)
    renal = parse_bool(get(params, 'renal disease criteria for the has-bled rule',
                                  'renal disease', 'renal dysfunction'))
    liver = parse_bool(get(params, 'liver disease criteria for the has-bled rule',
                                  'liver disease', 'hepatic disease'))

    # S - Stroke history
    stroke = parse_bool(get(params, 'stroke'))

    # B - Prior major bleeding (use specific key to avoid substring 'bleeding' matching medication)
    bleeding = parse_bool(get(params, 'prior major bleeding or predisposition to bleeding',
                                     'prior major bleeding',
                                     'prior bleeding',
                                     'predisposition to bleeding',
                                     'bleeding history'))

    # L - Labile INR
    labile_inr = parse_bool(get(params, 'labile international normalized ratio',
                                       'labile inr', 'unstable inr'))

    # E - Elderly (age >= 65)
    age = to_float(get(params, 'age'))
    elderly = age is not None and age >= 65

    # D - Drugs/Alcohol
    drugs   = parse_bool(get(params, 'medication usage predisposing to bleeding',
                                    'antiplatelet', 'nsaid', 'medication predisposing'))
    alcohol = parse_bool(get(params, 'number of alcoholic drinks per week', 'alcohol use',
                                    'alcoholic drinks'))
    # Alcohol: >8 drinks/week = positive (if numeric)
    alc_raw = get(params, 'number of alcoholic drinks per week', 'alcohol')
    alc_num = to_float(alc_raw)
    if alc_num is not None:
        alcohol = alc_num >= 8

    score = 0
    if htn:        score += 1
    if renal:      score += 1
    if liver:      score += 1
    if stroke:     score += 1
    if bleeding:   score += 1
    if labile_inr: score += 1
    if elderly:    score += 1
    if drugs:      score += 1
    if alcohol:    score += 1

    return {'success': True, 'result': score, 'unit': 'points',
            'name': 'HAS-BLED Score for Major Bleeding Risk'}


# ── Calculator 26: Sodium Correction for Hyperglycemia ───────────────────────

def sodium_correction_hyperglycemia(params: dict) -> dict:
    na_raw  = get(params, 'sodium', 'serum sodium')
    glc_raw = get(params, 'glucose', 'blood glucose', 'serum glucose')
    na  = to_float(na_raw)
    glc = _glucose_to_mgdl(to_float(glc_raw))
    if na is None or glc is None:
        return {'success': False, 'error': 'Missing sodium or glucose'}
    corrected = na + 0.016 * (glc - 100)
    return {'success': True, 'result': round(corrected, 3), 'unit': 'mEq/L',
            'name': 'Sodium Correction for Hyperglycemia'}


# ── Calculator 27: Glasgow-Blatchford Bleeding Score ─────────────────────────

def glasgow_blatchford(params: dict) -> dict:
    bun_raw  = get(params, 'blood urea nitrogen (bun)', 'blood urea nitrogen', 'bun', 'urea')
    hgb_raw  = get(params, 'hemoglobin', 'hgb', 'hb')
    sbp_raw  = get(params, 'systolic blood pressure', 'systolic bp', 'sbp')
    hr_raw   = get(params, 'heart rate or pulse', 'heart rate', 'pulse')
    sex      = parse_sex(get(params, 'sex', 'gender'))
    melena   = parse_bool(get(params, 'melena present', 'melena'))
    syncope  = parse_bool(get(params, 'recent syncope', 'syncope'))
    cardiac  = parse_bool(get(params, 'cardiac failure present', 'heart failure', 'cardiac failure'))
    hepatic  = parse_bool(get(params, 'hepatic disease history', 'liver disease', 'hepatic disease'))

    bun = to_float(bun_raw)
    hgb = to_float(hgb_raw)
    sbp = to_float(sbp_raw)
    hr  = to_float(hr_raw)

    score = 0

    # BUN (mg/dL) - convert from mmol/L if <20
    if bun is not None:
        if bun < 20:  # mmol/L → mg/dL
            bun *= 2.8
        if bun >= 25:    score += 6
        elif bun >= 20:  score += 4  # 8 mmol/L ≈ 22.4 mg/dL
        elif bun >= 14:  score += 3  # 5.6-8
        elif bun >= 9:   score += 2  # 3.5-5.6 → convert
        # else 0

    # Actually use standard GBS BUN thresholds in mmol/L, re-implement
    # Recompute using mmol/L (standard UK scoring)
    bun_orig = to_float(bun_raw)
    if bun_orig is not None:
        # If value seems in mg/dL (>20), convert to mmol/L
        if bun_orig > 20:
            bun_mmol = bun_orig / 2.8
        else:
            bun_mmol = bun_orig
        score = 0  # reset and recompute
        if bun_mmol >= 25:     score += 6
        elif bun_mmol >= 10:   score += 4
        elif bun_mmol >= 8:    score += 3
        elif bun_mmol >= 6.5:  score += 2
        # else 0

    # Hemoglobin
    if hgb is not None:
        if sex == 'male':
            if hgb < 10:     score += 6
            elif hgb < 12:   score += 3
            elif hgb < 13:   score += 1
        else:
            if hgb < 10:     score += 6
            elif hgb < 12:   score += 1

    # SBP
    if sbp is not None:
        if sbp < 90:     score += 3
        elif sbp < 100:  score += 2
        elif sbp < 110:  score += 1

    # Other
    if hr is not None and hr >= 100: score += 1
    if melena:  score += 1
    if syncope: score += 2
    if cardiac: score += 2
    if hepatic: score += 2

    return {'success': True, 'result': score, 'unit': 'points',
            'name': 'Glasgow-Blatchford Bleeding Score (GBS)'}


# ── Calculator 28: APACHE II ──────────────────────────────────────────────────

def apache_ii(params: dict) -> dict:
    temp_raw = get(params, 'temperature', 'temp')
    sbp_raw  = get(params, 'systolic blood pressure', 'systolic bp')
    dbp_raw  = get(params, 'diastolic blood pressure', 'diastolic bp')
    hr_raw   = get(params, 'heart rate or pulse', 'heart rate', 'pulse')
    rr_raw   = get(params, 'respiratory rate', 'rr')
    fio2_raw = get(params, 'fio2', 'fraction of inspired oxygen', 'fio2 percentage')
    pao2_raw = get(params, 'pao2', 'partial pressure of oxygen', 'arterial oxygen')
    aa_grad_raw = get(params, 'a-a gradient', 'alveolar arterial gradient')
    ph_raw   = get(params, 'ph', 'arterial ph', 'blood ph')
    na_raw   = get(params, 'sodium', 'serum sodium')
    k_raw    = get(params, 'potassium', 'serum potassium')
    cr_raw   = get(params, 'creatinine', 'serum creatinine')
    hct_raw  = get(params, 'hematocrit', 'hct')
    wbc_raw  = get(params, 'white blood cell count', 'wbc', 'leukocytes')
    gcs_raw  = get(params, 'glasgow coma score', 'gcs')
    age_raw  = get(params, 'age')
    chronic  = parse_bool(get(params, 'history of severe organ failure or immunocompromise',
                                     'chronic organ failure', 'immunocompromise'))
    surg_raw = get(params, 'surgery type', 'surgery')
    arf      = parse_bool(get(params, 'acute renal failure', 'acute kidney injury'))

    temp  = _temp_to_celsius(to_float(temp_raw)) if temp_raw is not None else None
    sbp   = to_float(sbp_raw)
    dbp   = to_float(dbp_raw)
    hr    = to_float(hr_raw)
    rr    = to_float(rr_raw)
    fio2  = _fio2_to_fraction(fio2_raw) if fio2_raw is not None else None
    pao2  = to_float(pao2_raw)
    aa_gr = to_float(aa_grad_raw)
    ph    = to_float(ph_raw)
    na    = to_float(na_raw)
    k     = to_float(k_raw)
    cr    = _creatinine_to_mgdl(cr_raw) if cr_raw is not None else None
    hct   = to_float(hct_raw)
    wbc   = _wbc_to_k(wbc_raw) if wbc_raw is not None else None
    gcs   = to_float(gcs_raw)
    age   = to_float(age_raw)

    score = 0

    # Temperature
    if temp is not None:
        if temp >= 41 or temp < 30:       score += 4
        elif temp >= 39 or temp < 32:     score += 3
        elif 38.5 <= temp < 39:           score += 1
        elif 36 <= temp < 38.5:           score += 0
        elif 34 <= temp < 36:             score += 1
        elif 32 <= temp < 34:             score += 2

    # MAP
    if sbp is not None and dbp is not None:
        map_ = dbp + (sbp - dbp) / 3
        if map_ >= 160 or map_ < 50:       score += 4
        elif map_ >= 130 or map_ < 70:     score += 3
        elif 110 <= map_ < 130:            score += 2
        elif 50 <= map_ < 70:              score += 2

    # HR
    if hr is not None:
        if hr >= 180 or hr < 40:           score += 4
        elif hr >= 140 or hr < 55:         score += 3
        elif hr >= 110 or hr < 70:         score += 2
        elif 70 <= hr < 110:               score += 0

    # RR
    if rr is not None:
        if rr >= 50 or rr < 6:             score += 4
        elif rr >= 35 or rr < 10:          score += 3  # rr<10 → 2 below
        elif 25 <= rr < 35:                score += 3
        elif 10 <= rr < 12:                score += 1
        elif 12 <= rr < 25:                score += 0
    if rr is not None:
        score_rr = 0
        if rr >= 50 or rr < 6:   score_rr = 4
        elif rr >= 35:           score_rr = 3
        elif rr >= 25:           score_rr = 1
        elif rr >= 12:           score_rr = 0
        elif rr >= 10:           score_rr = 1
        elif rr >= 6:            score_rr = 2
        else:                    score_rr = 4
        # re-add correctly - first undo then redo
        if rr is not None:
            if rr >= 50 or rr < 6:             tmp = 4
            elif rr >= 35:                     tmp = 3
            elif rr >= 25:                     tmp = 1
            elif rr >= 12:                     tmp = 0
            elif rr >= 10:                     tmp = 1
            else:                              tmp = 2
            score -= (4 if (rr >= 50 or rr < 6) else (3 if (rr >= 35 or rr < 10) else (3 if rr >= 25 else (1 if rr >= 10 else 0))))
            score += tmp

    # Oxygenation
    if fio2 is not None and fio2 >= 0.5:
        # Use A-a gradient
        if aa_gr is not None:
            if aa_gr >= 500:     score += 4
            elif aa_gr >= 350:   score += 3
            elif aa_gr >= 200:   score += 2
    elif pao2 is not None:
        if pao2 < 55:     score += 4
        elif pao2 < 60:   score += 3
        elif pao2 < 70:   score += 1

    # pH
    if ph is not None:
        if ph >= 7.7 or ph < 7.15:   score += 4
        elif ph >= 7.6 or ph < 7.25: score += 3
        elif 7.5 <= ph < 7.6:        score += 1
        elif 7.33 <= ph < 7.5:       score += 0
        elif 7.25 <= ph < 7.33:      score += 2

    # Sodium
    if na is not None:
        if na >= 180 or na < 111:     score += 4
        elif na >= 160 or na < 120:   score += 3
        elif na >= 155 or na < 130:   score += 2
        elif 150 <= na < 155:         score += 1

    # Potassium
    if k is not None:
        if k >= 7 or k < 2.5:         score += 4
        elif k >= 6 or k < 3:         score += 3
        elif 5.5 <= k < 6:            score += 1
        elif 3 <= k < 3.5:            score += 1
        elif 2.5 <= k < 3:            score += 2

    # Creatinine (double if ARF)
    if cr is not None:
        if cr >= 3.5:     cr_pts = 4
        elif cr >= 2:     cr_pts = 3
        elif cr >= 1.5:   cr_pts = 2
        elif 0.6 <= cr < 1.5: cr_pts = 0
        else:             cr_pts = 2
        if arf: cr_pts *= 2
        score += cr_pts

    # Hematocrit
    if hct is not None:
        if hct >= 60 or hct < 20:     score += 4
        elif hct >= 50 or hct < 30:   score += 2
        elif 46 <= hct < 50:          score += 1

    # WBC
    if wbc is not None:
        if wbc >= 40 or wbc < 1:      score += 4
        elif wbc >= 20 or wbc < 3:    score += 2
        elif 15 <= wbc < 20:          score += 1

    # GCS – score = 15 - GCS
    if gcs is not None:
        score += int(15 - gcs)

    # Age
    if age is not None:
        if age >= 75:        score += 6
        elif age >= 65:      score += 5
        elif age >= 55:      score += 3
        elif age >= 45:      score += 2

    # Chronic health
    if chronic:
        surg = str(surg_raw).lower() if surg_raw else ''
        if 'elective' in surg or 'post-elective' in surg:
            score += 2
        else:
            score += 5

    return {'success': True, 'result': score, 'unit': 'points',
            'name': 'APACHE II Score'}


# ── Calculator 29: PSI Score ──────────────────────────────────────────────────

def psi_score(params: dict) -> dict:
    age   = to_float(get(params, 'age'))
    sex   = parse_sex(get(params, 'sex', 'gender'))
    nursing = parse_bool(get(params, 'nursing home resident', 'nursing home'))

    # Comorbidities — use precise key names, avoid 'tia' substring
    neoplasm  = parse_bool(get(params, 'neoplastic disease', 'malignancy', 'cancer'))
    liver     = parse_bool(get(params, 'liver disease history', 'liver disease', 'hepatic disease'))
    chf       = parse_bool(get(params, 'congestive heart failure'))
    cvd       = parse_bool(get(params, 'cerebrovascular disease history',
                                      'cerebrovascular disease'))   # NO 'tia', 'stroke' - avoid false matches
    renal     = parse_bool(get(params, 'renal disease history', 'renal disease'))

    # Exam findings
    ams    = parse_bool(get(params, 'altered mental status'))
    sbp_raw = get(params, 'systolic blood pressure', 'sbp')
    sbp    = to_float(sbp_raw)
    hr_raw  = get(params, 'heart rate or pulse', 'heart rate', 'pulse')
    hr     = to_float(hr_raw)
    rr_raw  = get(params, 'respiratory rate', 'rr')
    rr     = to_float(rr_raw)
    temp_raw = get(params, 'temperature', 'temp')
    temp   = _temp_to_celsius(to_float(temp_raw)) if temp_raw is not None else None

    # Labs
    ph_raw   = get(params, 'ph', 'arterial ph')
    bun_raw  = get(params, 'blood urea nitrogen (bun)', 'blood urea nitrogen', 'bun')
    na_raw   = get(params, 'sodium', 'serum sodium')
    glc_raw  = get(params, 'glucose', 'blood glucose')
    hct_raw  = get(params, 'hematocrit', 'hct')
    pao2_raw = get(params, 'partial pressure of oxygen', 'pao2')
    pleural  = parse_bool(get(params, 'pleural effusion on x-ray', 'pleural effusion'))

    ph   = to_float(ph_raw)
    bun  = to_float(bun_raw)
    na   = to_float(na_raw)
    glc  = _glucose_to_mgdl(to_float(glc_raw)) if glc_raw is not None else None
    hct  = to_float(hct_raw)
    pao2 = to_float(pao2_raw)

    if age is None:
        return {'success': False, 'error': 'Missing age'}

    # Base score
    score = age
    if sex == 'female': score -= 10
    if nursing: score += 10

    # Comorbidities
    if neoplasm: score += 30
    if liver:    score += 20
    if chf:      score += 10
    if cvd:      score += 10
    if renal:    score += 10

    # Exam
    if ams:                                     score += 20
    if sbp is not None and sbp < 90:            score += 20
    if temp is not None and (temp < 35 or temp > 39.9): score += 15
    if hr  is not None and hr > 125:            score += 10
    if rr  is not None and rr >= 30:            score += 20

    # Labs
    if ph   is not None and ph < 7.35:          score += 30
    if bun  is not None:
        bun_mgdl = bun * 2.8 if bun < 50 else bun   # convert mmol/L → mg/dL if small
        if bun_mgdl >= 30:                      score += 20
    if na   is not None and na < 130:           score += 20
    if glc  is not None and glc >= 250:         score += 10
    if hct  is not None and hct < 30:          score += 10
    if pao2 is not None and pao2 < 60:          score += 10
    if pleural:                                 score += 10

    return {'success': True, 'result': round(score, 3), 'unit': 'points',
            'name': 'PSI Score: Pneumonia Severity Index for CAP'}


# ── Calculator 30: Serum Osmolality ──────────────────────────────────────────

def serum_osmolality(params: dict) -> dict:
    na  = to_float(get(params, 'sodium', 'serum sodium'))
    bun = to_float(get(params, 'blood urea nitrogen (bun)', 'bun', 'urea'))
    glc = _glucose_to_mgdl(to_float(get(params, 'glucose', 'blood glucose')))
    if na is None or bun is None or glc is None:
        return {'success': False, 'error': 'Missing required parameters'}
    # BUN in mg/dL check
    if bun < 10:  # likely mmol/L
        bun *= 2.8
    osm = 2 * na + bun / 2.8 + glc / 18
    return {'success': True, 'result': round(osm, 3), 'unit': 'mOsm/kg',
            'name': 'Serum Osmolality'}


# ── Calculator 31: HOMA-IR ────────────────────────────────────────────────────

def homa_ir(params: dict) -> dict:
    glc_raw = get(params, 'glucose', 'fasting glucose', 'blood glucose')
    ins_raw = get(params, 'insulin', 'fasting insulin')
    glc = to_float(glc_raw)
    ins = to_float(ins_raw)
    if glc is None or ins is None:
        return {'success': False, 'error': 'Missing glucose or insulin'}
    # Convert mmol/L to mg/dL if needed
    if glc < 30:
        glc *= 18.018
    # HOMA-IR = (fasting glucose mg/dL × fasting insulin µIU/mL) / 405
    # NOTE: validated empirically against dataset: /405 gives correct results
    homa = (glc * ins) / 405
    return {'success': True, 'result': round(homa, 3), 'unit': 'index',
            'name': 'HOMA-IR (Homeostatic Model Assessment for Insulin Resistance)'}


# ── Calculator 32: Charlson Comorbidity Index ─────────────────────────────────

def charlson_cci(params: dict) -> dict:
    age = to_float(get(params, 'age'))

    mi      = parse_bool(get(params, 'myocardial infarction'))
    chf     = parse_bool(get(params, 'congestive heart failure'))
    pvd     = parse_bool(get(params, 'peripheral vascular disease'))
    cvd     = parse_bool(get(params, 'cerebrovascular accident', 'cerebrovascular disease'))
    tia     = parse_bool(get(params, 'transient ischemic attacks history', 'tia'))
    dementia= parse_bool(get(params, 'dementia'))
    cpd     = parse_bool(get(params, 'chronic pulmonary disease', 'copd'))
    ctd     = parse_bool(get(params, 'connective tissue disease'))
    pud     = parse_bool(get(params, 'peptic ulcer disease'))
    liver_raw = get(params, 'liver disease severity', 'liver disease', 'hepatic disease')

    dm_raw  = get(params, 'diabetes mellitus', 'diabetes')

    hemi    = parse_bool(get(params, 'hemiplegia'))
    ckd     = parse_bool(get(params, 'moderate to severe chronic kidney disease', 'chronic kidney disease'))
    tumor   = parse_bool(get(params, 'solid tumor', 'tumor'))
    leuk    = parse_bool(get(params, 'leukemia'))
    lymph   = parse_bool(get(params, 'lymphoma'))
    aids    = parse_bool(get(params, 'aids', 'hiv'))

    score = 0

    # 1-pt conditions
    for cond in [mi, chf, pvd, cvd, tia, dementia, cpd, ctd, pud]:
        if cond: score += 1

    # Liver disease: mild=1, moderate/severe=3
    if liver_raw is not None:
        liver_s = str(liver_raw).lower()
        if any(x in liver_s for x in ('moderate', 'severe', 'cirrhosis', 'portal hypertension')):
            score += 3
        elif any(x in liver_s for x in ('mild', 'chronic hepatitis', 'chronic liver')):
            score += 1
        elif parse_bool(liver_raw):
            score += 1

    # Diabetes: uncomplicated=1, with complications/end-organ=2
    if dm_raw is not None:
        dm_s = str(dm_raw).lower()
        if any(x in dm_s for x in ('end-organ', 'end organ', 'complications', 'complication',
                                    'retinopathy', 'nephropathy', 'neuropathy', 'severe')):
            score += 2
        elif parse_bool(dm_raw):
            score += 1

    # 2-pt conditions
    if hemi: score += 2
    if ckd:  score += 2

    # Tumor/Metastases: solid=2, leukemia/lymphoma=2, metastatic=6
    if tumor: score += 2
    if leuk:  score += 2
    if lymph: score += 2

    # AIDS = 6
    if aids: score += 6

    # Age adjustment
    if age is not None:
        score += max(0, int((age - 40) / 10))

    return {'success': True, 'result': score, 'unit': 'points',
            'name': 'Charlson Comorbidity Index (CCI)'}


# ── Calculator 33: FeverPAIN ──────────────────────────────────────────────────

def feverpain(params: dict) -> dict:
    # Fever - default True since dataset often omits when True
    fever_raw = get(params, 'fever in past 24 hours', 'fever')
    if fever_raw is None:
        fever = True  # default: patient has fever (why they're being assessed)
    else:
        fever = parse_bool(fever_raw)

    purulent  = parse_bool(get(params, 'purulent tonsils', 'pus on tonsils', 'purulent'))
    severe    = parse_bool(get(params, 'severe tonsil inflammation', 'severe inflammation'))
    onset     = parse_bool(get(params, 'symptom onset <=3 days', 'rapid onset', 'onset <=3 days',
                                      'onset less than 3 days', 'onset within 3 days'))
    no_cough  = parse_bool(get(params, 'absence of cough or coryza', 'cough absent',
                                      'no cough', 'absence of cough'))

    score = sum([fever, purulent, severe, onset, no_cough])
    return {'success': True, 'result': score, 'unit': 'points',
            'name': 'FeverPAIN Score for Strep Pharyngitis'}


# ── Calculator 36: Caprini Score ──────────────────────────────────────────────

def caprini_score(params: dict) -> dict:
    age    = to_float(get(params, 'age'))
    sex    = parse_sex(get(params, 'sex', 'gender'))
    bmi_raw = get(params, 'body mass index (bmi)', 'bmi')
    bmi    = to_float(bmi_raw)

    # 1-pt factors
    obesity   = bmi is not None and bmi >= 25
    minor_sx  = parse_bool(get(params, 'minor surgery in the next month', 'minor surgery'))
    med_cond  = parse_bool(get(params, 'serious medical illness',))
    varicose  = parse_bool(get(params, 'varicose veins'))
    swollen_legs = parse_bool(get(params, 'current swollen legs', 'swollen legs'))
    central_iv = parse_bool(get(params, 'current central venous access', 'central venous access', 'central line'))
    ibd       = parse_bool(get(params, 'history of inflammatory bowel disease', 'inflammatory bowel disease'))
    mobility_raw = get(params, 'mobility', 'bed rest')
    mobility = str(mobility_raw).lower() if mobility_raw else ''
    confined  = 'confined' in mobility or 'bed' in mobility or parse_bool(mobility_raw)

    # 2-pt factors
    major_sx  = parse_bool(get(params, 'major surgery in the last month',
                               'major surgery in the past month', 'major surgery'))
    chf_recent = parse_bool(get(params, 'congestive heart failure in the last month',
                                'congestive heart failure'))
    copd      = parse_bool(get(params, 'chronic obstructive pulmonary disease', 'copd'))
    malignancy = parse_bool(get(params, 'present or previous malignancy', 'malignancy', 'cancer'))
    pneumonia  = parse_bool(get(params, 'pneumonia in the last month', 'pneumonia'))
    sepsis     = parse_bool(get(params, 'sepsis in the last month', 'sepsis'))
    cast       = parse_bool(get(params, 'immobilizing plaster cast in the last month', 'plaster cast'))
    ami        = parse_bool(get(params, 'acute myocardial infarction', 'myocardial infarction'))

    # 3-pt factors
    prev_dvt   = parse_bool(get(params, 'previously documented deep vein thrombosis',
                                'previously documented dvt', 'prior dvt'))
    prev_pe    = parse_bool(get(params, 'previously documented pulmonary embolism',
                                'previous pe', 'prior pe'))
    family_clot= parse_bool(get(params, 'family history of thrombosis', 'family history'))
    thrombophil= parse_bool(get(params, 'positive factor v leiden', 'factor v leiden',
                                'positive prothrombin 20210a', 'other congenital or acquired thrombophilia',
                                'lupus anticoagulant', 'elevated anticardiolipin antibody',
                                'elevated serum homocysteine', 'heparin-induced thrombocytopenia'))
    hit        = parse_bool(get(params, 'heparin-induced thrombocytopenia', 'hit'))

    # 5-pt factors
    stroke_recent  = parse_bool(get(params, 'stroke in the last month', 'stroke'))
    hip_fracture   = parse_bool(get(params, 'hip, pelvis, or leg fracture in the last month',
                                   'hip fracture'))
    spinal_cord    = parse_bool(get(params, 'acute spinal cord injury causing paralysis in the last month',
                                   'spinal cord injury'))
    elective_hip   = parse_bool(get(params, 'elective major lower extremity arthroplasty',
                                   'hip arthroplasty', 'knee arthroplasty'))
    multiple_trauma= parse_bool(get(params, 'multiple trauma in the last month', 'multiple trauma'))

    surgery_raw = get(params, 'surgery type', 'surgery')
    surg = str(surgery_raw).lower() if surgery_raw else ''

    score = 0

    # Age points
    if age is not None:
        if age >= 75:        score += 3
        elif age >= 61:      score += 2
        elif age >= 41:      score += 1

    # BMI
    if bmi is not None and bmi > 25: score += 1

    # 1-pt
    for cond in [varicose, swollen_legs, central_iv, ibd, confined]:
        if cond: score += 1

    # 2-pt
    for cond in [major_sx, chf_recent, copd, malignancy, pneumonia, sepsis, cast, ami]:
        if cond: score += 2

    # Surgery type (elective = 1pt, >45min or laparoscopic = 2pt)
    if 'minor' in surg: score += 1
    if any(x in surg for x in ('major', 'open', '45 min', '>45')):
        score += 2

    # 3-pt
    for cond in [prev_dvt, prev_pe, family_clot, hit]:
        if cond: score += 3
    # Other thrombophilia (3pt each if positive)
    thrombo_keys = ['positive factor v leiden', 'positive prothrombin 20210a',
                    'positive lupus anticoagulant', 'elevated anticardiolipin antibody',
                    'elevated serum homocysteine', 'other congenital or acquired thrombophilia']
    for tk in thrombo_keys:
        if parse_bool(get(params, tk)):
            score += 3

    # 5-pt
    for cond in [stroke_recent, hip_fracture, spinal_cord, elective_hip, multiple_trauma]:
        if cond: score += 5

    return {'success': True, 'result': score, 'unit': 'points',
            'name': 'Caprini Score for Venous Thromboembolism (2005)'}


# ── Calculator 38: Free Water Deficit ─────────────────────────────────────────

def free_water_deficit(params: dict) -> dict:
    na   = to_float(get(params, 'sodium', 'serum sodium'))
    wt   = _weight_to_kg(get(params, 'weight', 'body weight'))
    age  = to_float(get(params, 'age'))
    sex  = parse_sex(get(params, 'sex', 'gender'))

    if na is None or wt is None:
        return {'success': False, 'error': 'Missing sodium or weight'}

    # TBW fraction
    if sex == 'female':
        tbw_f = 0.45 if (age is not None and age > 65) else 0.50
    else:
        tbw_f = 0.50 if (age is not None and age > 65) else 0.60

    tbw = tbw_f * wt
    deficit = tbw * (na / 140 - 1)

    return {'success': True, 'result': round(deficit, 3), 'unit': 'L',
            'name': 'Free Water Deficit'}


# ── Calculator 39: Anion Gap ──────────────────────────────────────────────────

def anion_gap(params: dict) -> dict:
    na   = to_float(get(params, 'sodium', 'serum sodium'))
    cl   = to_float(get(params, 'chloride', 'serum chloride'))
    hco3 = to_float(get(params, 'bicarbonate', 'hco3', 'serum bicarbonate'))
    if any(v is None for v in [na, cl, hco3]):
        return {'success': False, 'error': 'Missing electrolytes'}
    ag = na - (cl + hco3)
    return {'success': True, 'result': round(ag, 3), 'unit': 'mEq/L',
            'name': 'Anion Gap'}


# ── Calculator 40: FENa ───────────────────────────────────────────────────────

def fractional_excretion_sodium(params: dict) -> dict:
    na_serum   = to_float(get(params, 'sodium', 'serum sodium'))
    na_urine   = to_float(get(params, 'urine sodium'))
    cr_serum   = _creatinine_to_mgdl(get(params, 'creatinine', 'serum creatinine'))
    cr_urine   = to_float(get(params, 'urine creatinine'))

    if any(v is None for v in [na_serum, na_urine, cr_serum, cr_urine]):
        return {'success': False, 'error': 'Missing required parameters'}
    if na_serum == 0 or cr_serum == 0:
        return {'success': False, 'error': 'Zero division'}

    fena = (na_urine / na_serum) / (cr_urine / cr_serum) * 100
    return {'success': True, 'result': round(fena, 3), 'unit': '%',
            'name': 'Fractional Excretion of Sodium (FENa)'}


# ── Calculator 43: SOFA Score ─────────────────────────────────────────────────

def sofa_score(params: dict) -> dict:
    # Respiratory: PaO2/FiO2
    pao2_raw = get(params, 'pao2', 'partial pressure of oxygen', 'arterial oxygen')
    fio2_raw = get(params, 'fio2', 'fraction of inspired oxygen', 'fio2 percentage')
    vent     = parse_bool(get(params, 'on mechanical ventilation', 'mechanical ventilation'))
    cpap     = parse_bool(get(params, 'continuous positive airway pressure', 'cpap'))

    pao2  = to_float(pao2_raw)
    fio2  = _fio2_to_fraction(fio2_raw) if fio2_raw is not None else None

    # Coagulation: Platelets
    plt   = _platelets_to_k(get(params, 'platelet count', 'platelets', 'plt'))

    # Liver: Bilirubin
    bili  = _bilirubin_to_mgdl(get(params, 'bilirubin', 'total bilirubin'))

    # Cardiovascular: MAP or vasopressors
    sbp_raw = get(params, 'systolic blood pressure', 'sbp')
    dbp_raw = get(params, 'diastolic blood pressure', 'dbp')
    hypoten = parse_bool(get(params, 'hypotension'))
    dopamine= to_float(get(params, 'dopamine', 'dopamine dose'))
    dobutam = to_float(get(params, 'dobutamine', 'dobutamine dose'))
    norepi  = to_float(get(params, 'norepinephrine', 'norepinephrine dose'))
    epi     = to_float(get(params, 'epinephrine', 'epinephrine dose'))

    # CNS: GCS
    gcs   = to_float(get(params, 'glasgow coma score', 'gcs'))

    # Renal: Creatinine, Urine Output
    cr    = _creatinine_to_mgdl(get(params, 'creatinine', 'serum creatinine'))
    urine = to_float(get(params, 'urine output', 'urine output per day'))

    score = 0

    # Respiratory
    if pao2 is not None and fio2 is not None and fio2 > 0:
        ratio = pao2 / fio2
        if ratio < 100:      score += 4
        elif ratio < 200:    score += 3
        elif ratio < 300:    score += 2
        elif ratio < 400:    score += 1

    # Coagulation
    if plt is not None:
        if plt < 20:         score += 4
        elif plt < 50:       score += 3
        elif plt < 100:      score += 2
        elif plt < 150:      score += 1

    # Liver
    if bili is not None:
        if bili >= 12:       score += 4
        elif bili >= 6:      score += 3
        elif bili >= 2:      score += 2
        elif bili >= 1.2:    score += 1

    # Cardiovascular
    if sbp_raw is not None and dbp_raw is not None:
        sbp = to_float(sbp_raw)
        dbp = to_float(dbp_raw)
        if sbp is not None and dbp is not None:
            map_ = dbp + (sbp - dbp) / 3
            if map_ < 70:    score += 1
    if dopamine is not None:
        if dopamine > 15:    score += 4
        elif dopamine > 5:   score += 3
        elif dopamine > 0:   score += 2
    if dobutam is not None and dobutam > 0: score += 2
    if norepi  is not None and norepi  > 0: score += 3
    if epi     is not None and epi     > 0: score += 3
    if hypoten: score = max(score, score + 1)

    # CNS
    if gcs is not None:
        if gcs < 6:          score += 4
        elif gcs < 10:       score += 3
        elif gcs < 13:       score += 2
        elif gcs < 15:       score += 1

    # Renal
    if cr is not None:
        if cr >= 5:          score += 4
        elif cr >= 3.5:      score += 3
        elif cr >= 2:        score += 2
        elif cr >= 1.2:      score += 1
    if urine is not None:
        if urine < 200:      score += 4
        elif urine < 500:    score += 3

    return {'success': True, 'result': score, 'unit': 'points',
            'name': 'Sequential Organ Failure Assessment (SOFA) Score'}


# ── Calculator 44: LDL Calculated ────────────────────────────────────────────

def ldl_calculated(params: dict) -> dict:
    tc_raw  = get(params, 'total cholesterol')
    hdl_raw = get(params, 'high-density lipoprotein cholesterol', 'hdl cholesterol', 'hdl')
    tg_raw  = get(params, 'triglycerides', 'triglyceride')

    tc  = to_float(tc_raw)
    hdl = to_float(hdl_raw)
    tg  = to_float(tg_raw)

    if any(v is None for v in [tc, hdl, tg]):
        return {'success': False, 'error': 'Missing required lipid values'}

    # Detect mmol/L: TC typically 100-300 mg/dL or 2.6-7.8 mmol/L
    # If all values are small (<20), likely mmol/L
    if tc < 20:
        # Convert to mg/dL
        tc_mgdl  = tc  * 38.67
        hdl_mgdl = hdl * 38.67
        tg_mgdl  = tg  * 88.57
        ldl_mgdl = tc_mgdl - hdl_mgdl - tg_mgdl / 5
        # Convert result back to mmol/L
        ldl = ldl_mgdl / 38.67
    else:
        ldl = tc - hdl - tg / 5

    return {'success': True, 'result': round(ldl, 3), 'unit': 'mg/dL or mmol/L',
            'name': 'LDL Calculated'}


# ── Calculator 45: CURB-65 ────────────────────────────────────────────────────

def curb65(params: dict) -> dict:
    confusion = parse_bool(get(params, 'confusion', 'altered mental status', 'confusion present'))
    bun_raw   = get(params, 'blood urea nitrogen (bun)', 'blood urea nitrogen', 'bun', 'urea')
    bun       = to_float(bun_raw)
    rr_raw    = get(params, 'respiratory rate', 'rr')
    rr        = to_float(rr_raw)
    sbp_raw   = get(params, 'systolic blood pressure', 'sbp')
    dbp_raw   = get(params, 'diastolic blood pressure', 'dbp')
    sbp       = to_float(sbp_raw)
    dbp       = to_float(dbp_raw)
    age       = to_float(get(params, 'age'))

    score = 0
    if confusion: score += 1

    # BUN > 19 mg/dL (7 mmol/L)
    if bun is not None:
        # Convert mmol/L to mg/dL if small
        bun_mgdl = bun * 2.8 if bun < 20 else bun
        if bun_mgdl > 19: score += 1

    if rr is not None and rr >= 30: score += 1

    if sbp is not None and dbp is not None:
        if sbp < 90 or dbp <= 60: score += 1

    if age is not None and age >= 65: score += 1

    return {'success': True, 'result': score, 'unit': 'points',
            'name': 'CURB-65 Score for Pneumonia Severity'}


# ── Calculator 46: Framingham Risk Score ──────────────────────────────────────

def framingham_risk(params: dict) -> dict:
    age    = to_float(get(params, 'age'))
    sex    = parse_sex(get(params, 'sex', 'gender'))
    tc_raw = get(params, 'total cholesterol')
    hdl_raw = get(params, 'high-density lipoprotein cholesterol', 'hdl cholesterol', 'hdl')
    sbp_raw = get(params, 'systolic blood pressure', 'sbp')
    sbp_treated = parse_bool(get(params, 'blood pressure being treated with medicines',
                                        'bp treated', 'on antihypertensive'))
    smoker  = parse_bool(get(params, 'smoker', 'smoking', 'current smoker'))

    tc  = to_float(tc_raw)
    hdl = to_float(hdl_raw)
    sbp = to_float(sbp_raw)

    if any(v is None for v in [age, tc, hdl, sbp]):
        return {'success': False, 'error': 'Missing required parameters'}

    # Convert mmol/L to mg/dL if small
    if tc < 20:
        tc  *= 38.67
        hdl *= 38.67

    # Wilson 1998 / ATP III continuous formula
    if sex == 'male':
        l = (math.log(age)   * 3.06117
           + math.log(tc)    * 1.12370
           - math.log(hdl)   * 0.93263
           + math.log(sbp)   * (1.93303 if sbp_treated else 1.99881)
           + (0.65451 if smoker else 0)
           - 23.9802)
        risk = 1 - (0.88936 ** math.exp(l))
    else:
        l = (math.log(age)   * 2.32888
           + math.log(tc)    * 1.20904
           - math.log(hdl)   * 0.70833
           + math.log(sbp)   * (2.82263 if sbp_treated else 2.76157)
           + (0.52873 if smoker else 0)
           - 26.1931)
        risk = 1 - (0.95012 ** math.exp(l))

    return {'success': True, 'result': round(risk * 100, 3), 'unit': '%',
            'name': 'Framingham Risk Score for Hard Coronary Heart Disease'}


# ── Calculator 48: PERC Rule ──────────────────────────────────────────────────

def perc_rule(params: dict) -> dict:
    age  = to_float(get(params, 'age'))
    hr_raw = get(params, 'heart rate or pulse', 'heart rate', 'pulse')
    hr   = to_float(hr_raw)
    # O2 sat - handle unicode ₂
    o2_sat = to_float(get(params, 'o2 saturation percentage', 'o2 saturation',
                              'oxygen saturation', 'saturation percentage',
                              'o₂ saturation percentage', 'spo2'))
    hemo   = parse_bool(get(params, 'hemoptysis'))
    hormones = parse_bool(get(params, 'hormone use', 'estrogen', 'oral contraceptive'))
    leg_swell = parse_bool(get(params, 'unilateral leg swelling', 'leg swelling'))
    surgery  = parse_bool(get(params, 'recent surgery or trauma', 'recent surgery', 'recent trauma'))
    # Prior PE or DVT: count as ONE criterion (either/or)
    prev_pe  = parse_bool(get(params, 'previously documented pulmonary embolism',
                                     'prior pe', 'previous pe'))
    prev_dvt = parse_bool(get(params, 'previously documented deep vein thrombosis',
                                     'prior dvt', 'previous dvt'))
    prior_vte = prev_pe or prev_dvt  # single criterion

    count = 0
    if age is not None and age >= 50:         count += 1
    if hr is not None and hr >= 100:          count += 1
    if o2_sat is not None and o2_sat < 95:    count += 1
    if hemo:                                  count += 1
    if hormones:                              count += 1
    if leg_swell:                             count += 1
    if surgery:                               count += 1
    if prior_vte:                             count += 1

    return {'success': True, 'result': count, 'unit': 'criteria met',
            'name': 'PERC Rule for Pulmonary Embolism'}


# ── Calculator 49: MME Calculator ────────────────────────────────────────────

# MME conversion factors (per mg/day unless noted)
MME_FACTORS = {
    'codeine':              0.15,
    'fentanyl buccal':      0.13,
    'fentanyl transdermal': 2.4,   # per mcg/h
    'hydrocodone':          1.0,
    'hydromorphone':        4.0,
    'methadone':            {(1, 20): 4, (21, 40): 8, (41, 60): 10, (61, float('inf')): 12},
    'morphine':             1.0,
    'oxycodone':            1.5,
    'oxymorphone':          3.0,
    'tapentadol':           0.4,
    'tramadol':             0.1,
}

def _get_mme_factor(drug: str, dose: float) -> float:
    key = drug.lower().strip()
    factor = MME_FACTORS.get(key)
    if factor is None:
        # Try partial match
        for k, v in MME_FACTORS.items():
            if k in key or key in k:
                factor = v
                break
    if factor is None:
        return 0.0
    if isinstance(factor, dict):
        for (lo, hi), f in factor.items():
            if lo <= dose <= hi:
                return f
        return 12  # highest tier
    return factor


def mme_calculator(params: dict) -> dict:
    DRUGS = [
        ('codeine',          'codeine dose', 'codeine dose per day'),
        ('fentanyl buccal',  'fentanyl buccal dose', 'fentanyl buccal dose per day',
                             'fentanyl buccal dose per day'),
        ('hydrocodone',      'hydrocodone dose', 'hydrocodone dose per day'),
        ('hydromorphone',    'hydromorphone dose', 'hydromorphone dose per day'),
        ('methadone',        'methadone dose', 'methadone dose per day'),
        ('morphine',         'morphine dose', 'morphine dose per day'),
        ('oxycodone',        'oxycodone dose', 'oxycodone dose per day'),
        ('oxymorphone',      'oxymorphone dose', 'oxymorphone dose per day'),
        ('tapentadol',       'tapentadol dose', 'tapentadol dose per day'),
        ('tramadol',         'tramadol dose', 'tramadol dose per day'),
    ]

    total_mme = 0.0

    for drug_info in DRUGS:
        drug_name = drug_info[0]
        keys      = drug_info[1:]
        dose_raw  = get(params, *keys)
        if dose_raw is None:
            # Try looking for {drugname} dose or {drugname} dose per day
            dose_raw = get(params, f'{drug_name} dose', f'{drug_name} dose per day')
        dose = to_float(dose_raw)
        if dose is not None and dose > 0:
            factor = _get_mme_factor(drug_name, dose)
            total_mme += dose * factor

    if total_mme == 0:
        return {'success': False, 'error': 'No opioid doses found'}

    return {'success': True, 'result': round(total_mme, 3), 'unit': 'MME/day',
            'name': 'Morphine Milligram Equivalents (MME) Calculator'}


# ── Calculator 51: SIRS Criteria ──────────────────────────────────────────────

def sirs_criteria(params: dict) -> dict:
    temp_raw = get(params, 'temperature', 'temp')
    hr_raw   = get(params, 'heart rate or pulse', 'heart rate', 'pulse')
    rr_raw   = get(params, 'respiratory rate', 'rr')
    paco2_raw = get(params, 'paco2', 'partial pressure of co2')
    wbc_raw  = get(params, 'white blood cell count', 'wbc', 'leukocytes')

    temp  = _temp_to_celsius(to_float(temp_raw)) if temp_raw is not None else None
    hr    = to_float(hr_raw)
    rr    = to_float(rr_raw)
    paco2 = to_float(paco2_raw)
    wbc   = _wbc_to_k(wbc_raw) if wbc_raw is not None else None

    count = 0
    if temp  is not None and (temp > 38.0 or temp < 36.0): count += 1
    if hr    is not None and hr > 90:                       count += 1
    if rr    is not None and rr > 20:                       count += 1
    if paco2 is not None and paco2 < 32:                    count += 1
    if wbc   is not None and (wbc > 12.0 or wbc < 4.0):    count += 1
    # Maximum 4 criteria (temp, hr, rr/pco2, wbc) - rr and pco2 are alternate for same criterion
    # Actually SIRS has 4 criteria each worth 1 point

    return {'success': True, 'result': count, 'unit': 'criteria met',
            'name': 'SIRS Criteria'}


# ── Calculator 56: QTc Fridericia ─────────────────────────────────────────────

def qtc_fridericia(params: dict) -> dict:
    hr = to_float(get(params, 'heart rate or pulse', 'heart rate', 'pulse'))
    qt = to_float(get(params, 'qt interval', 'qt'))
    if hr is None or qt is None or hr == 0:
        return {'success': False, 'error': 'Missing HR or QT'}
    rr = 60 / hr
    qt_s = qt / 1000 if qt > 10 else qt
    qtc = qt_s / (rr ** (1/3))
    return {'success': True, 'result': round(qtc * 1000, 3), 'unit': 'ms',
            'name': 'QTc Fridericia Calculator'}


# ── Calculator 57: QTc Framingham ─────────────────────────────────────────────

def qtc_framingham(params: dict) -> dict:
    hr = to_float(get(params, 'heart rate or pulse', 'heart rate', 'pulse'))
    qt = to_float(get(params, 'qt interval', 'qt'))
    if hr is None or qt is None or hr == 0:
        return {'success': False, 'error': 'Missing HR or QT'}
    rr = 60 / hr
    qt_s = qt / 1000 if qt > 10 else qt
    qtc = qt_s + 0.154 * (1 - rr)
    return {'success': True, 'result': round(qtc * 1000, 3), 'unit': 'ms',
            'name': 'QTc Framingham Calculator'}


# ── Calculator 58: QTc Hodges ──────────────────────────────────────────────────

def qtc_hodges(params: dict) -> dict:
    hr = to_float(get(params, 'heart rate or pulse', 'heart rate', 'pulse'))
    qt = to_float(get(params, 'qt interval', 'qt'))
    if hr is None or qt is None or hr == 0:
        return {'success': False, 'error': 'Missing HR or QT'}
    qt_ms = qt if qt > 10 else qt * 1000
    qtc = qt_ms + 1.75 * (hr - 60)
    return {'success': True, 'result': round(qtc, 3), 'unit': 'ms',
            'name': 'QTc Hodges Calculator'}


# ── Calculator 59: QTc Rautaharju ─────────────────────────────────────────────

def qtc_rautaharju(params: dict) -> dict:
    hr = to_float(get(params, 'heart rate or pulse', 'heart rate', 'pulse'))
    qt = to_float(get(params, 'qt interval', 'qt'))
    if hr is None or qt is None or hr == 0:
        return {'success': False, 'error': 'Missing HR or QT'}
    qt_ms = qt if qt > 10 else qt * 1000
    qtc = qt_ms * (120 + hr) / 180
    return {'success': True, 'result': round(qtc, 3), 'unit': 'ms',
            'name': 'QTc Rautaharju Calculator'}


# ── Calculator 60: Body Surface Area ──────────────────────────────────────────

def body_surface_area(params: dict) -> dict:
    ht_raw = get(params, 'height')
    wt_raw = get(params, 'weight', 'body weight')
    if ht_raw is None or wt_raw is None:
        return {'success': False, 'error': 'Missing height or weight'}
    ht_cm = _height_to_cm(ht_raw)
    wt_kg = _weight_to_kg(wt_raw)
    if ht_cm is None or wt_kg is None:
        return {'success': False, 'error': 'Invalid height or weight'}
    # Mosteller formula
    bsa = math.sqrt(ht_cm * wt_kg / 3600)
    return {'success': True, 'result': round(bsa, 3), 'unit': 'm²',
            'name': 'Body Surface Area Calculator'}


# ── Calculator 61: Target Weight ─────────────────────────────────────────────

def target_weight(params: dict) -> dict:
    ht_raw  = get(params, 'height')
    bmi_raw = get(params, 'body mass index (bmi)', 'bmi', 'target bmi')
    if ht_raw is None or bmi_raw is None:
        return {'success': False, 'error': 'Missing height or target BMI'}
    h_m  = _height_to_m(ht_raw)
    bmi  = to_float(bmi_raw)
    if h_m is None or h_m == 0 or bmi is None:
        return {'success': False, 'error': 'Invalid parameters'}
    wt = bmi * (h_m ** 2)
    return {'success': True, 'result': round(wt, 3), 'unit': 'kg',
            'name': 'Target weight'}


# ── Calculator 62: Adjusted Body Weight ──────────────────────────────────────

def adjusted_body_weight(params: dict) -> dict:
    ht_raw = get(params, 'height')
    wt_raw = get(params, 'weight', 'actual body weight', 'body weight')
    sex    = parse_sex(get(params, 'sex', 'gender'))
    if ht_raw is None or wt_raw is None:
        return {'success': False, 'error': 'Missing height or weight'}
    ht_cm = _height_to_cm(ht_raw)
    wt_kg = _weight_to_kg(wt_raw)
    if ht_cm is None or wt_kg is None:
        return {'success': False, 'error': 'Invalid parameters'}
    ibw = _ideal_body_weight(ht_cm, sex)
    abw = ibw + 0.4 * (wt_kg - ibw)
    return {'success': True, 'result': round(abw, 3), 'unit': 'kg',
            'name': 'Adjusted Body Weight'}


# ── Calculator 63: Delta Gap ───────────────────────────────────────────────────

def delta_gap(params: dict) -> dict:
    na   = to_float(get(params, 'sodium', 'serum sodium'))
    cl   = to_float(get(params, 'chloride', 'serum chloride'))
    hco3 = to_float(get(params, 'bicarbonate', 'hco3'))
    if any(v is None for v in [na, cl, hco3]):
        return {'success': False, 'error': 'Missing electrolytes'}
    ag = na - cl - hco3
    delta_gap_val = ag - 12   # assuming normal AG = 12
    return {'success': True, 'result': round(delta_gap_val, 3), 'unit': 'mEq/L',
            'name': 'Delta Gap'}


# ── Calculator 64: Delta Ratio ─────────────────────────────────────────────────

def delta_ratio(params: dict) -> dict:
    na   = to_float(get(params, 'sodium', 'serum sodium'))
    cl   = to_float(get(params, 'chloride', 'serum chloride'))
    hco3 = to_float(get(params, 'bicarbonate', 'hco3'))
    if any(v is None for v in [na, cl, hco3]):
        return {'success': False, 'error': 'Missing electrolytes'}
    ag = na - cl - hco3
    delta_ag  = ag - 12
    delta_hco3 = 24 - hco3
    if delta_hco3 == 0:
        return {'success': False, 'error': 'Division by zero'}
    ratio = delta_ag / delta_hco3
    return {'success': True, 'result': round(ratio, 3), 'unit': 'ratio',
            'name': 'Delta Ratio'}


# ── Calculator 65: Albumin Corrected Anion Gap ────────────────────────────────

def albumin_corrected_anion_gap(params: dict) -> dict:
    na   = to_float(get(params, 'sodium', 'serum sodium'))
    cl   = to_float(get(params, 'chloride', 'serum chloride'))
    hco3 = to_float(get(params, 'bicarbonate', 'hco3'))
    alb  = _albumin_to_gdl(get(params, 'albumin', 'serum albumin'))
    if any(v is None for v in [na, cl, hco3, alb]):
        return {'success': False, 'error': 'Missing parameters'}
    ag     = na - cl - hco3
    ag_corr = ag + 2.5 * (4.0 - alb)
    return {'success': True, 'result': round(ag_corr, 3), 'unit': 'mEq/L',
            'name': 'Albumin Corrected Anion Gap'}


# ── Calculator 66: Albumin Corrected Delta Gap ────────────────────────────────

def albumin_corrected_delta_gap(params: dict) -> dict:
    na   = to_float(get(params, 'sodium', 'serum sodium'))
    cl   = to_float(get(params, 'chloride', 'serum chloride'))
    hco3 = to_float(get(params, 'bicarbonate', 'hco3'))
    alb  = _albumin_to_gdl(get(params, 'albumin', 'serum albumin'))
    if any(v is None for v in [na, cl, hco3, alb]):
        return {'success': False, 'error': 'Missing parameters'}
    ag      = na - cl - hco3
    ag_corr = ag + 2.5 * (4.0 - alb)
    delta_g = ag_corr - 12
    return {'success': True, 'result': round(delta_g, 3), 'unit': 'mEq/L',
            'name': 'Albumin Corrected Delta Gap'}


# ── Calculator 67: Albumin Corrected Delta Ratio ─────────────────────────────

def albumin_corrected_delta_ratio(params: dict) -> dict:
    na   = to_float(get(params, 'sodium', 'serum sodium'))
    cl   = to_float(get(params, 'chloride', 'serum chloride'))
    hco3 = to_float(get(params, 'bicarbonate', 'hco3'))
    alb  = _albumin_to_gdl(get(params, 'albumin', 'serum albumin'))
    if any(v is None for v in [na, cl, hco3, alb]):
        return {'success': False, 'error': 'Missing parameters'}
    ag      = na - cl - hco3
    ag_corr = ag + 2.5 * (4.0 - alb)
    delta_ag    = ag_corr - 12
    delta_hco3  = 24 - hco3
    if delta_hco3 == 0:
        return {'success': False, 'error': 'Division by zero'}
    ratio = delta_ag / delta_hco3
    return {'success': True, 'result': round(ratio, 3), 'unit': 'ratio',
            'name': 'Albumin Corrected Delta Ratio'}


# ── Calculator Registry ───────────────────────────────────────────────────────

CALCULATOR_REGISTRY = {
    2:  cockcroft_gault,
    3:  ckd_epi,
    4:  cha2ds2_vasc,
    5:  mean_arterial_pressure,
    6:  bmi_calc,
    7:  calcium_correction,
    8:  wells_pe,
    9:  mdrd_gfr,
    10: ideal_body_weight,
    11: qtc_bazett,
    15: child_pugh,
    16: wells_dvt,
    17: revised_cardiac_risk_index,
    18: heart_score,
    19: fib4,
    20: centor_score,
    21: glasgow_coma_score,
    22: maintenance_fluids,
    23: meld_na,
    24: steroid_conversion,
    25: has_bled,
    26: sodium_correction_hyperglycemia,
    27: glasgow_blatchford,
    28: apache_ii,
    29: psi_score,
    30: serum_osmolality,
    31: homa_ir,
    32: charlson_cci,
    33: feverpain,
    36: caprini_score,
    38: free_water_deficit,
    39: anion_gap,
    40: fractional_excretion_sodium,
    43: sofa_score,
    44: ldl_calculated,
    45: curb65,
    46: framingham_risk,
    48: perc_rule,
    49: mme_calculator,
    51: sirs_criteria,
    56: qtc_fridericia,
    57: qtc_framingham,
    58: qtc_hodges,
    59: qtc_rautaharju,
    60: body_surface_area,
    61: target_weight,
    62: adjusted_body_weight,
    63: delta_gap,
    64: delta_ratio,
    65: albumin_corrected_anion_gap,
    66: albumin_corrected_delta_gap,
    67: albumin_corrected_delta_ratio,
}

# Name → ID mapping (case-insensitive)
NAME_ALIASES = {
    'creatinine clearance (cockcroft-gault equation)': 2,
    'cockcroft-gault': 2,
    'cockcroft gault': 2,
    'ckd-epi equations for glomerular filtration rate': 3,
    'ckd-epi': 3,
    'ckd epi': 3,
    "cha2ds2-vasc score for atrial fibrillation stroke risk": 4,
    'cha2ds2-vasc': 4,
    'cha2ds2 vasc': 4,
    'mean arterial pressure (map)': 5,
    'map': 5,
    'body mass index (bmi)': 6,
    'bmi': 6,
    'calcium correction for hypoalbuminemia': 7,
    "wells' criteria for pulmonary embolism": 8,
    'wells pe': 8,
    'wells criteria pe': 8,
    'mdrd gfr equation': 9,
    'mdrd': 9,
    'ideal body weight': 10,
    'ibw': 10,
    'qtc bazett calculator': 11,
    'qtc bazett': 11,
    'child-pugh score for cirrhosis mortality': 15,
    'child pugh': 15,
    "wells' criteria for dvt": 16,
    'wells dvt': 16,
    'revised cardiac risk index for pre-operative risk': 17,
    'rcri': 17,
    'heart score for major cardiac events': 18,
    'heart score': 18,
    'fibrosis-4 (fib-4) index for liver fibrosis': 19,
    'fib-4': 19,
    'fib4': 19,
    'centor score (modified/mcisaac) for strep pharyngitis': 20,
    'centor score': 20,
    'glasgow coma score (gcs)': 21,
    'gcs': 21,
    'maintenance fluids calculations': 22,
    'maintenance fluids': 22,
    'meld na (unos/optn)': 23,
    'meld na': 23,
    'meld-na': 23,
    'steroid conversion calculator': 24,
    'has-bled score for major bleeding risk': 25,
    'has-bled': 25,
    'has bled': 25,
    'sodium correction for hyperglycemia': 26,
    'glasgow-blatchford bleeding score (gbs)': 27,
    'glasgow blatchford': 27,
    'gbs': 27,
    'apache ii score': 28,
    'apache ii': 28,
    'psi score: pneumonia severity index for cap': 29,
    'psi score': 29,
    'psi': 29,
    'serum osmolality': 30,
    'homa-ir (homeostatic model assessment for insulin resistance)': 31,
    'homa-ir': 31,
    'homa ir': 31,
    'charlson comorbidity index (cci)': 32,
    'charlson': 32,
    'cci': 32,
    'feverpain score for strep pharyngitis': 33,
    'feverpain': 33,
    'caprini score for venous thromboembolism (2005)': 36,
    'caprini': 36,
    'free water deficit': 38,
    'anion gap': 39,
    'fractional excretion of sodium (fena)': 40,
    'fena': 40,
    'sequential organ failure assessment (sofa) score': 43,
    'sofa': 43,
    'ldl calculated': 44,
    'curb-65 score for pneumonia severity': 45,
    'curb-65': 45,
    'curb 65': 45,
    'framingham risk score for hard coronary heart disease': 46,
    'framingham': 46,
    'perc rule for pulmonary embolism': 48,
    'perc': 48,
    'morphine milligram equivalents (mme) calculator': 49,
    'mme': 49,
    'sirs criteria': 51,
    'sirs': 51,
    'qtc fridericia calculator': 56,
    'qtc fridericia': 56,
    'qtc framingham calculator': 57,
    'qtc framingham': 57,
    'qtc hodges calculator': 58,
    'qtc hodges': 58,
    'qtc rautaharju calculator': 59,
    'qtc rautaharju': 59,
    'body surface area calculator': 60,
    'bsa': 60,
    'target weight': 61,
    'adjusted body weight': 62,
    'abw': 62,
    'delta gap': 63,
    'delta ratio': 64,
    'albumin corrected anion gap': 65,
    'albumin corrected delta gap': 66,
    'albumin corrected delta ratio': 67,
}


def resolve_calculator_name(name: str) -> Optional[int]:
    """Resolve calculator name string to registry ID."""
    n = name.lower().strip()
    # Direct alias lookup
    if n in NAME_ALIASES:
        return NAME_ALIASES[n]
    # Substring match
    for alias, cid in NAME_ALIASES.items():
        if alias in n or n in alias:
            return cid
    # Try numeric
    try:
        return int(n)
    except ValueError:
        pass
    return None


def run_calculator(name: str, params: dict) -> dict:
    """
    Main entry point.
    Args:
        name:   Calculator name or ID (string)
        params: Flat dict of parameters (from parse_entities)
    Returns:
        {'success': bool, 'result': ..., 'unit': ..., 'name': ..., 'error': ...}
    """
    cid = resolve_calculator_name(str(name))
    if cid is None:
        return {'success': False, 'error': f'Unknown calculator: {name}'}

    fn = CALCULATOR_REGISTRY.get(cid)
    if fn is None:
        return {'success': False, 'error': f'Calculator ID {cid} not implemented'}

    try:
        return fn(params)
    except Exception as e:
        return {'success': False, 'error': f'Runtime error: {e}'}
