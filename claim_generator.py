"""
Three new data entries are generated for each original data entry:

- true: Entities are correct + Calculation result is correct

- partially true: Entities are correct + Calculation result is incorrect (±20% perturbation)

- false: Entities are incorrect (±20% perturbation) + Calculation result is incorrect

The output name in the Claim uses the Output Name column from medcalc_cal_np.csv.

"""

import pandas as pd
import ast
import random
import re
from datetime import datetime, timedelta

random.seed(42)

# Load the Calculator ID and Output Name 

def load_output_name_map(cal_csv_path):
    df = pd.read_csv(cal_csv_path, encoding='latin1')
    # Remove any leading or trailing spaces from the Output Name.
    df['Output Name'] = df['Output Name'].str.strip()
    return dict(zip(df['Calculator ID'].astype(int), df['Output Name']))


# Rule-based Calculator IDs

RULE_BASED_CALC_IDS = {4, 8, 15, 16, 17, 18, 20, 21, 25, 27, 29, 32, 33, 43, 45, 48, 51, 68, 69}


# Output unit of each calculator

CALC_UNITS = {
    2:  "mL/min",               # Creatinine Clearance (Cockcroft-Gault)
    3:  "mL/min/1.73 m²",       # CKD-EPI GFR
    4:  "",                     # CHA2DS2-VASc Score
    5:  "mm Hg",                # Mean Arterial Pressure
    6:  "kg/m²",                # BMI
    7:  "mg/dL",                # Calcium Correction for Hypoalbuminemia
    8:  "",                     # Wells' Criteria for PE
    9:  "mL/min/1.73 m²",       # MDRD GFR
    10: "kg",                   # Ideal Body Weight
    11: "msec",                 # QTc Bazett
    13: "",                     # Estimated Due Date (date)
    15: "",                     # Child-Pugh Score
    16: "",                     # Wells' Criteria for DVT
    17: "",                     # Revised Cardiac Risk Index
    18: "",                     # HEART Score
    19: "",                     # FIB-4 Index
    20: "",                     # Centor Score
    21: "",                     # Glasgow Coma Score
    22: "mL/hr",                # Maintenance Fluids
    23: "",                     # MELD Na
    24: "mg",                   # Steroid Conversion
    25: "",                     # HAS-BLED Score
    26: "mEq/L",                # Sodium Correction for Hyperglycemia
    27: "",                     # Glasgow-Blatchford Score
    28: "",                     # APACHE II Score
    29: "",                     # PSI Score
    30: "mOsm/kg",              # Serum Osmolality
    31: "",                     # HOMA-IR
    32: "",                     # Charlson Comorbidity Index
    33: "",                     # FeverPAIN Score
    36: "",                     # Caprini Score
    38: "L",                    # Free Water Deficit
    39: "mEq/L",                # Anion Gap
    40: "%",                    # FENa
    43: "",                     # SOFA Score
    44: "mg/dL",                # LDL Calculated
    45: "",                     # CURB-65 Score
    46: "%",                    # Framingham Risk Score
    48: "",                     # PERC Rule
    49: "MME/day",              # Morphine Milligram Equivalents
    51: "",                     # SIRS Criteria
    56: "msec",                 # QTc Fridericia
    57: "msec",                 # QTc Framingham
    58: "msec",                 # QTc Hodges
    59: "msec",                 # QTc Rautaharju
    60: "m²",                   # Body Surface Area
    61: "kg",                   # Target Weight
    62: "kg",                   # Adjusted Body Weight
    63: "mEq/L",                # Delta Gap
    64: "",                     # Delta Ratio
    65: "mEq/L",                # Albumin Corrected Anion Gap
    66: "mEq/L",                # Albumin Corrected Delta Gap
    67: "",                     # Albumin Corrected Delta Ratio
    68: "",                     # Estimated Date of Conception (date)
    69: "",                     # Estimated Gestational Age (date/tuple)
}


# Possible values ​​for categorical entities

CATEGORY_OPTIONS = {
    'sex': ['Female', 'Male'],
    'Race': ['Hispanic', 'White', 'Asian', 'Black'],
    'Ascites': ['absent', 'moderate', 'slight'],
    'Encephalopathy': ['Grade 1-2', 'Grade 3-4','No Encephalopathy'],
    'Suspicion History': ['Highly suspicious', 'Moderately suspicious', 'Slightly suspicious'],
    'Electrocardiogram Test': [
        'Non-specific repolarization disturbance', 'Normal', 'Significant ST deviation',
    ],
    'Initial troponin': [
        'between the normal limit or up to three times the normal limit',
        'greater than three times normal limit',
        'less than or equal to normal limit',
    ],
    'Liver disease severity': ['mild', 'none', 'moderate to severe'],
    'Solid tumor': ['localized', 'metastatic', 'none'],
    'Diabetes mellitus': ['end-organ damage', 'none or diet-controlled', 'uncomplicated'],
    'Best eye response': [
        'eye opening to pain', 'eye opening to verbal command',
        'eyes open spontaneously', 'no eye opening',
    ],
    'Best verbal response': [
        'confused', 'inappropriate words', 'incomprehensible sounds',
        'no verbal response', 'oriented',
    ],
    'Best motor response': [
        'extension to pain', 'flexion to pain', 'localizes pain',
        'no motor response', 'obeys commands', 'withdrawal from pain',
    ],
    'Surgery Type': [
        'Elective', 'Emergency', 'Nonoperative', 'arthroscopic',
        'elective major lower extremity arthroplasty', 'laparoscopic', 'major', 'minor',
    ],
    'Mobility': ['confined to bed >72 hours', 'normal', 'on bed rest'],
    'target steroid': [
        'Betamethasone IV', 'Cortisone PO', 'Dexamethasone PO',
        'Hydrocortisone IV', 'Hydrocortisone PO', 'MethylPrednisoLONE PO',
        'PredniSONE PO', 'PrednisoLONE PO', 'Triamcinolone IV',
    ],
}


# Perturbation function

def perturb_numeric(value, lower, upper, factor=0.20, max_tries=20):
    for _ in range(max_tries):
        direction = random.choice([-1, 1])
        ratio = 1 + direction * random.uniform(0.12, factor)
        new_val = round(value * ratio, 4)
        try:
            if float(lower) <= new_val <= float(upper):
                continue
        except (ValueError, TypeError):
            pass
        if new_val != value:
            return new_val
    return round(value * 1.25, 4)


def perturb_date(date_str, fmt='%m/%d/%Y', days_range=(30, 180)):
    try:
        dt = datetime.strptime(date_str, fmt)
        delta = random.choice([-1, 1]) * random.randint(*days_range)
        return (dt + timedelta(days=delta)).strftime(fmt)
    except Exception:
        return date_str


def perturb_category(key, value):
    others = [o for o in CATEGORY_OPTIONS.get(key, []) if o != value]
    return random.choice(others) if others else value


def perturb_entity_value(key, value, lower=None, upper=None):
    if isinstance(value, list) and len(value) >= 2 and isinstance(value[0], (int, float)):
        raw = value[0]
        new_val = perturb_numeric(raw, lower, upper)
        if isinstance(raw, bool):
            new_val = round(new_val, 2)
        elif isinstance(raw, int):
            new_val = int(round(new_val))
            if new_val == raw:
                new_val = raw + random.choice([-2, -1, 1, 2])
        else:
            decimals = min(len(str(raw).rstrip('0').split('.')[-1]) if '.' in str(raw) else 0, 2)
            new_val = round(new_val, decimals)
        return [new_val, value[1]]
    elif isinstance(value, bool):
        return not value
    elif isinstance(value, float):
        return round(perturb_numeric(value, lower, upper), 4)
    elif isinstance(value, int):
        new_val = int(round(perturb_numeric(value, lower, upper)))
        if new_val == value:
            new_val = value + random.choice([-2, -1, 1, 2])
        return new_val
    elif isinstance(value, str) and '/' in value and len(value) == 10:
        try:
            datetime.strptime(value, '%m/%d/%Y')
            return perturb_date(value)
        except ValueError:
            pass
        return value
    elif isinstance(value, str):
        return perturb_category(key, value)
    return value


# Answer formatting (including units)

def _num_str(val, output_type):
    try:
        f = float(val)
        if output_type == 'integer':
            return str(int(round(f)))
        return str(round(f, 4))
    except (ValueError, TypeError):
        return str(val)


def format_answer_with_unit(answer, output_type, calc_id):
    unit = CALC_UNITS.get(int(calc_id), "")
    if output_type == 'date':
        return str(answer)
    num = _num_str(answer, output_type)
    return f"{num} {unit}".strip() if unit else num


def perturb_answer_with_unit(answer, lower, upper, output_type, calc_id):
    unit = CALC_UNITS.get(int(calc_id), "")
    if output_type == 'date':
        ans_str = str(answer)
        if ans_str.startswith('('):
            return f"('{random.randint(1,5)} weeks', '{random.randint(0,6)} days')"
        return perturb_date(ans_str)
    try:
        val = float(answer)
        new_val = perturb_numeric(val, lower, upper)
        if output_type == 'integer':
            new_val = int(round(new_val))
            if new_val == int(val):
                new_val = int(val) + random.choice([-2, -1, 1, 2])
            num = str(new_val)
        else:
            num = str(round(new_val, 4))
        return f"{num} {unit}".strip() if unit else num
    except (ValueError, TypeError):
        return str(answer)



# Entity value formatting


def format_entity_value(key, value):
    if isinstance(value, list) and len(value) >= 2:
        return f"{value[0]} {value[1]}"
    elif isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


# Claim build: using output_name

def build_claim(entities_dict, answer_str, output_name):
    """
    output_name: from medcalc_cal_np.csv
    """
    parts = [f"the {k} is {format_entity_value(k, v)}" for k, v in entities_dict.items()]
    if parts:
        return (f"Based on the patient's data where {', '.join(parts)}, "
                f"the {output_name} is {answer_str}.")
    return f"The {output_name} is {answer_str}."


# Explanation generation

def find_snippet_in_note(note, value_str, window=120):
    note_lower = note.lower()
    val_lower = value_str.lower().strip()
    if not val_lower:
        return None
    idx = note_lower.find(val_lower)
    if idx == -1:
        num_only = re.sub(r'[^\d.]', '', val_lower)
        if num_only:
            idx = note_lower.find(num_only)
    if idx == -1:
        return None
    start = max(0, idx - 40)
    end = min(len(note), idx + window)
    snippet = note[start:end].replace('\n', ' ').strip()
    return f'"{snippet}"'


def entity_value_to_search_str(value):
    if isinstance(value, list) and len(value) >= 2:
        return str(value[0])
    elif isinstance(value, bool):
        return ""
    return str(value)


def build_entity_explanation(key, true_value, claim_value, note, is_correct):
    true_fmt  = format_entity_value(key, true_value)
    claim_fmt = format_entity_value(key, claim_value)
    snippet = find_snippet_in_note(note, entity_value_to_search_str(true_value))
    note_ref = f' The patient note states {snippet}.' if snippet else \
               ' This information is documented in the patient note.'
    if is_correct:
        return (f'The claim states the {key} is {claim_fmt}, which is correct.'
                f'{note_ref}')
    else:
        return (f'The claim states the {key} is {claim_fmt}, but this is incorrect.'
                f' According to the patient note, the {key} is {true_fmt}.{note_ref}')


def build_calculation_explanation(claim_answer_str, gt_answer, gt_explanation,
                                  calc_name, output_type, calc_id, is_calc_correct):
    gt_fmt = format_answer_with_unit(gt_answer, output_type, calc_id)
   
    if output_type == 'date' or calc_id in RULE_BASED_CALC_IDS:
        if is_calc_correct:
            return (f'The calculation result in the claim is {claim_answer_str}. '
                    f'According to the {calc_name} formula, the claim\'s answer matches the correct result of {gt_fmt}, '
                    f'so the calculation is correct.')
        else:
            return (f'The calculation result in the claim is {claim_answer_str}. '
                    f'According to the {calc_name} formula, the correct answer should be {gt_fmt}, but the claim states {claim_answer_str}, '
                    f'which does not match the correct result, so the calculation is incorrect.')

    try:
        gt_val = float(gt_answer)
        num_m = re.match(r'([\-\d.]+)', claim_answer_str.strip())
        claim_val = float(num_m.group(1)) if num_m else None
        pct_diff = abs(claim_val - gt_val) / abs(gt_val) * 100 \
                   if (claim_val is not None and gt_val != 0) else None
    except (ValueError, TypeError):
        gt_val = claim_val = pct_diff = None

    if is_calc_correct:
        pct_str = f' (difference: {pct_diff:.2f}%)' if pct_diff is not None else ''
        return (f'The calculation result in the claim is {claim_answer_str}. '
                f'According to the {calc_name} formula, the correct answer is {gt_fmt}. '
                f'The claim\'s answer is within 5% of the correct result{pct_str}, '
                f'so the calculation is correct.')
    else:
        pct_str = f' (difference: {pct_diff:.2f}%, exceeding the 5% threshold)' \
                  if pct_diff is not None else ''
        return (f'The calculation result in the claim is {claim_answer_str}. '
                f'According to the {calc_name} formula, the correct answer is {gt_fmt}. '
                f'The claim\'s answer deviates from the correct result{pct_str}, '
                f'so the calculation is incorrect.')


def build_explanation(true_entities, claim_entities, claim_str,
                      gt_answer, gt_explanation, calc_name, output_name,
                      output_type, calc_id, note, label):
    is_true           = (label == 'true')
    is_partially_true = (label == 'partially true')

    lines = []

    # Extractive Claim Verification
    lines.append('Extractive Claim Verification:')
    for key, true_val in true_entities.items():
        claim_val = claim_entities.get(key, true_val)
        entity_correct = is_true or is_partially_true
        lines.append('- ' + build_entity_explanation(
            key, true_val, claim_val, note, entity_correct))

    # Implicit Calculation Claim Verification
    lines.append('')
    lines.append('Implicit Calculation Claim Verification:')

    pattern = rf'the {re.escape(output_name)} is (.+?)\s*\.$'
    m = re.search(pattern, claim_str, re.IGNORECASE)
    claim_answer_str = m.group(1).strip() if m else '(unknown)'

    calc_correct = is_true
    lines.append('- Conclusion: ' + build_calculation_explanation(
        claim_answer_str, gt_answer, gt_explanation,
        calc_name, output_type, calc_id, calc_correct
    ))

    lines.append(f'- Ground Truth Explanation: {gt_explanation}')

    return '\n'.join(lines)

   

# Transformation Logic

def process_row(row, output_name_map):
    try:
        entities = ast.literal_eval(row['Relevant Entities'])
    except Exception:
        return []

    calc_id        = row['Calculator ID']
    calc_name      = row['Calculator Name']         
    output_name    = output_name_map.get(int(calc_id), calc_name) 
    output_type    = row['Output Type']
    gt_answer      = row['Ground Truth Answer']
    gt_explanation = row['Ground Truth Explanation']
    lower          = row['Lower Limit']
    upper          = row['Upper Limit']
    note           = row['Patient Note']

    true_ans  = format_answer_with_unit(gt_answer, output_type, calc_id)
    wrong_ans = perturb_answer_with_unit(gt_answer, lower, upper, output_type, calc_id)

    perturbed_ents = {k: perturb_entity_value(k, v, lower, upper)
                      for k, v in entities.items()}

    claim_true    = build_claim(entities,       true_ans,  output_name)
    claim_partial = build_claim(entities,       wrong_ans, output_name)
    claim_false   = build_claim(perturbed_ents, wrong_ans, output_name)

    expl_true = build_explanation(
        entities, entities, claim_true,
        gt_answer, gt_explanation, calc_name, output_name,
        output_type, calc_id, note, 'true'
    )
    expl_partial = build_explanation(
        entities, entities, claim_partial,
        gt_answer, gt_explanation, calc_name, output_name,
        output_type, calc_id, note, 'partially true'
    )
    expl_false = build_explanation(
        entities, perturbed_ents, claim_false,
        gt_answer, gt_explanation, calc_name, output_name,
        output_type, calc_id, note, 'false'
    )

    base = {
        'Row Number':          row['Row Number'],
        'Calculator ID':       calc_id,
        'Calculator Name':     calc_name,
        'Output Name':         output_name,
        'Category':            row['Category'],
        'Output Type':         output_type,
        'Note ID':             row['Note ID'],
        'Note Type':           row['Note Type'],
        'Evidence':            note,
        'Relevant Entities':   row['Relevant Entities'],
        'Ground Truth Answer': gt_answer,
        'Lower Limit':         lower,
        'Upper Limit':         upper,
    }

    return [
        {**base, 'Claim': claim_true,    'Label': 'true',           'Explanation': expl_true},
        {**base, 'Claim': claim_partial, 'Label': 'partially true', 'Explanation': expl_partial},
        {**base, 'Claim': claim_false,   'Label': 'false',          'Explanation': expl_false},
    ]


# Output

def main(input_path, output_path, cal_csv_path):
    output_name_map = load_output_name_map(cal_csv_path)
    print(f"Loaded {len(output_name_map)} output name mappings")

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    all_records, errors = [], 0
    for _, row in df.iterrows():
        records = process_row(row, output_name_map)
        if records:
            all_records.extend(records)
        else:
            errors += 1

    df_new = (pd.DataFrame(all_records)
                .sample(frac=1, random_state=42)
                .reset_index(drop=True))
    df_new.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\nSaved {len(df_new)} rows → {output_path}")
    print(f"Skipped: {errors}")
    print("\nLabel distribution:")
    print(df_new['Label'].value_counts())
    print("\nSample claim vs output_name check:")
    for cid in [2, 3, 9, 11, 56, 31, 46]:
        row = df_new[(df_new['Calculator ID'] == cid) & (df_new['Label'] == 'true')].iloc[0]
        print(f"  [{cid}] output_name='{row['Output Name']}' | claim tail: ...{row['Claim'][-80:]}")
    return df_new


if __name__ == '__main__':
    main(
        input_path='medcalc_train_full.csv',
        output_path='medcalc_train_claim_full.csv',
        cal_csv_path='medcalc_cal_np.csv',
    )


