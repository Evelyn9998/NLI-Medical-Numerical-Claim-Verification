"""
Few-shot examples for the LLM-as-judge error classifier.
=========================================================
Source : 12 samples where both annotators agreed (kappa-validated).
         Explanations are based on both annotators' real written annotations
         from EA-WT-few-shot-sample.docx and EA-Evelyn-few-shot-sample.docx.
         Binary labels reflect the agreed labels from both xlsx annotation files.

Sample allocation
-----------------
Type 1 – failed_parameter_extraction          : idx 77, 38
Type 2 – verification_logic_error             : idx 151, 11, 36
Type 3 – incorrect_formula_criteria           : idx 57, 32, 23
Type 4 – computation_error                    : idx 84, 54
Type 5 – omitted_calculation_process_or_result: idx 2
Type 6 – other_error                          : idx 14
"""

FEW_SHOT_EXAMPLES = [

    # ── Type 1: failed_parameter_extraction | idx=77 ──────────────────────────
    {
        "index": 77,
        "primary_type": "failed parameter extraction",
        "true_label": "partially true",
        "predicted_label": "false",
        "claim": "Based on the patient's data where the age is 39 years, the Liver disease severity is none, the Moderate to severe Chronic Kidney Disease is no, the CCI Score is -2.",
        "evidence": "In March 2014, a 39-year-old Korean professional male soldier presented with a 6-month history of various nonspecific symptoms. He received hospital treatment and underwent several health examinations; however, no specific findings were observed. The symptoms were dizziness, fatigue, asthenia, irritability, elevated blood pressure (above systolic 160 mmHg), palpitation, eyestrain, and tinnitus, which aggravated with time. The patient had an unremarkable medical history and was a current smoker (",
        "model_reasoning": "The provided evidence does not mention liver disease severity, moderate to severe chronic kidney disease, or CCI Score. The claim is based on information not present in the evidence. The evidence focuses on the patient's symptoms, medical history, and diagnosis of lead poisoning. Therefore, the claim cannot be evaluated based on the provided evidence.",
        "annotation": {
            "failed_parameter_extraction": 1,
            "verification_logic_error": 0,
            "incorrect_formula_criteria": 0,
            "computation_error": 0,
            "omitted_calculation_process_or_result": 1,
            "other_error": 0,
            "explanation": (
                "failed parameter extraction: The model failed to extract the parameter 'age=39' "
                "from both the claim and evidence, and failed to apply the default dataset assumption "
                "that variables absent from both claim and evidence should be treated as false/none "
                "(not as missing/unverifiable). The LLM incorrectly treated absence of 'liver disease "
                "severity' and 'chronic kidney disease' in the evidence as grounds to reject the claim. "
                "omitted_calculation_process_or_result: The failure in parameter extraction caused the "
                "model to skip the CCI score calculation entirely."
            ),
        },
    },

    # ── Type 1: failed_parameter_extraction | idx=38 ──────────────────────────
    {
        "index": 38,
        "primary_type": "failed parameter extraction",
        "true_label": "partially true",
        "predicted_label": "false",
        "claim": "Based on the patient's data where the age is 45 years, the Suspicion History is Moderately suspicious, the HEART Score is 4.",
        "evidence": "A 45-year-old white female with a long-standing history of uncontrolled gastroesophageal reflux disease presented for laparoscopic Nissen fundoplication. The patient had a 14-year history of cold urticaria in addition to osteoarthritis, diverticulitis, and restless leg syndrome. Prior triggers for the patient's cold urticaria included consuming ice chips, being in cold environments and colonoscopy bowel preparation. She was also allergic to naproxen, guaifenesin, and menthol.",
        "model_reasoning": "The claim does not provide any information about the patient's data, Suspicion History, or HEART Score. However, the provided patient data does not mention anything about the HEART Score or Suspicion History. It does mention the patient's age (45 years) and a history of various medical conditions, but it does not provide enough information to determine the Suspicion History or HEART Score. Therefore, the claim cannot be verified as true or false based on the provided evidence.",
        "annotation": {
            "failed_parameter_extraction": 1,
            "verification_logic_error": 1,
            "incorrect_formula_criteria": 0,
            "computation_error": 0,
            "omitted_calculation_process_or_result": 0,
            "other_error": 0,
            "explanation": (
                "failed parameter extraction: The model lacks the medical domain knowledge required "
                "to classify Suspicion History as 'Moderately suspicious'. GERD is a classic cardiac "
                "mimic — it can cause chest discomfort that is neither clearly cardiac nor clearly "
                "non-cardiac — which places this patient in the 'Moderately suspicious' category of "
                "the HEART Score. The LLM failed to extract this parameter correctly and instead "
                "treated it as unverifiable. "
                "verification_logic_error: Because the LLM could not extract Suspicion History, it "
                "concluded the claim was entirely unverifiable and output 'false', rather than "
                "recognising that the available parameters were correctly stated."
            ),
        },
    },

    # ── Type 2: verification_logic_error | idx=151 ────────────────────────────
    {
        "index": 151,
        "primary_type": "verification logic errors",
        "true_label": "false",
        "predicted_label": "partially true",
        "claim": "Based on the patient's data where the Current Date is 11/16/2007, the Last menstrual date is 02/02/2008, the Estimated Gestational Age is ('4 weeks', '1 days').",
        "evidence": "The patient's last menstrual period was on 10/16/2007. Today's date is 01/24/2008.",
        "model_reasoning": "The claim's Current Date (11/16/2007) is incorrect, as the actual Current Date is 01/24/2008. However, the Last menstrual date (02/02/2008) is close to the actual Last menstrual date (10/16/2007). The Estimated Gestational Age is calculated based on the Last menstrual date, and 02/02/2008 minus 10/16/2007 is approximately 4 weeks and 1 day, which matches the claim. Therefore, the claim is partially true due to the incorrect Current Date.",
        "annotation": {
            "failed_parameter_extraction": 0,
            "verification_logic_error": 1,
            "incorrect_formula_criteria": 0,
            "computation_error": 0,
            "omitted_calculation_process_or_result": 0,
            "other_error": 0,
            "explanation": (
                "verification_logic_error: Both input entities in the claim are wrong — the Current "
                "Date (11/16/2007 vs actual 01/24/2008) and the Last menstrual date (02/02/2008 vs "
                "actual 10/16/2007). Under dataset rules, any incorrect input entity means the label "
                "must be 'false'. The LLM instead applied 'partially true' because the approximate "
                "date difference happened to match the gestational age claim. "
                "The LLM correctly identified that both input entities were wrong, "
                "but failed to apply the labelling rule: any incorrect input entity "
                "must yield 'false', regardless of whether the derived result appears "
                "to coincidentally match."
            ),
        },
    },

    # ── Type 2: verification_logic_error | idx=11 ─────────────────────────────
    {
        "index": 11,
        "primary_type": "verification logic errors",
        "true_label": "partially true",
        "predicted_label": "false",
        "claim": "Based on the patient's data where the Diastolic Blood Pressure is 75.0 mm hg, the Systolic Blood Pressure is 120.0 mm hg, the Mean Arterial Pressure is 72.5478 mm Hg.",
        "evidence": "A 21-year-old man presented to our emergency room with a complaint of right lower abdominal pain for 2 days. On physical examination, the vital signs were as follows: temperature, 37.3°C; heart rate, 94/min; respiratory rate, 20/min; blood pressure, 120/75 mm Hg.",
        "model_reasoning": "To calculate the Mean Arterial Pressure (MAP), we need to use the formula: MAP = (2 * Diastolic Blood Pressure + Systolic Blood Pressure) / 3. Given the Diastolic Blood Pressure is 75.0 mm Hg and the Systolic Blood Pressure is 120.0 mm Hg, we can plug these values into the formula: MAP = (2 * 75.0 + 120.0) / 3 = (150 + 120) / 3 = 270 / 3 = 90. Therefore, the Mean Arterial Pressure is 90.0 mm Hg, not 72.5478 mm Hg.",
        "annotation": {
            "failed_parameter_extraction": 0,
            "verification_logic_error": 1,
            "incorrect_formula_criteria": 0,
            "computation_error": 0,
            "omitted_calculation_process_or_result": 0,
            "other_error": 0,
            "explanation": (
                "verification_logic_error: Both input parameters (DBP=75.0, SBP=120.0) are correctly "
                "extracted. The LLM computes MAP=90 using the standard formula, sees it differs from "
                "the claim value of 72.5478, and labels the claim 'false'. However, since all input "
                "entities are correct and only the final result differs, the correct label is "
                "'partially true' — the LLM correctly extracted all input parameters but failed "
                "to apply this labelling rule."
            ),
        },
    },

    # ── Type 2: verification_logic_error | idx=36 ─────────────────────────────
    {
        "index": 36,
        "primary_type": "verification logic errors",
        "true_label": "true",
        "predicted_label": "false",
        "claim": "Based on the patient's data where the Pre-operative creatinine is 0.7 mg/dL, the Elevated-risk surgery is yes, the age is 51 years, the Cardiac Risk Index Score is 1.",
        "evidence": "A 51-year-old female visited us due to asymptomatic macroscopic hematuria persisting for two weeks. She had experienced discomfort, not pain, in the stomach. She had been followed up at another clinic for polycythemia for several years. Cystoscopy revealed a nonpapillary tumor about 2 cm in diameter, emanating from the left ureteral orifice (). Blood results at presentation showed Hb 14.8 g/dL, Cr 0.7 mg/dL, estimated GFR 68.1 mL/min, WBC 6.6 x 103/uL, CRP 5.35 mg/dL, LDH 184 IU/L, and Ca2+ 9.9 mg/dL.",
        "model_reasoning": "The claim states that the Elevated-risk surgery is yes, based on the provided data. The Cardiac Risk Index Score is 1, which is generally considered low risk. However, the patient's age (51 years) is not typically considered high risk for cardiac complications. The pre-operative creatinine level (0.7 mg/dL) is within normal limits, which suggests that the patient's kidney function is not significantly impaired. The American Society of Anesthesiologists (ASA) Physical Status Classification System is often used to assess the risk of surgery, and a score of 1 is considered normal.",
        "annotation": {
            "failed_parameter_extraction": 0,
            "verification_logic_error": 1,
            "incorrect_formula_criteria": 0,
            "computation_error": 0,
            "omitted_calculation_process_or_result": 0,
            "other_error": 0,
            "explanation": (
                "verification_logic_error: All input entities are correctly extractable from the "
                "evidence. The LLM made a qualitative judgment based on 'likely' risk and labelled "
                "the claim 'false' without applying the RCRI formula — failing to follow the "
                "prescribed verification process of extracting and confirming each input parameter "
                "before calculating the score."
            ),
        },
    },

    # ── Type 3: incorrect_formula_criteria | idx=57 ───────────────────────────
    {
        "index": 57,
        "primary_type": "incorrect formula & criteria application",
        "true_label": "true",
        "predicted_label": "false",
        "claim": "Based on the patient's data where the input steroid is Cortisone PO 260.997, the target steroid is Betamethasone IV, the Equivalent Steroid Dosage is 7.8299 mg.",
        "evidence": "Patient has taken 260.997 mg of Cortisone PO.",
        "model_reasoning": "To verify the claim, we need to know the conversion factor between Cortisone and Betamethasone. The conversion factor is approximately 1.5-1.6 mg of Betamethasone to 1 mg of Cortisone. Given the patient's dose of 260.997 mg of Cortisone, we can calculate the equivalent dose of Betamethasone. Assuming a conversion factor of 1.55 (midpoint of the range), the equivalent dose of Betamethasone would be 260.997 mg * 1.55 = 404.47 mg. This is significantly different from the claimed 7.8299 mg.",
        "annotation": {
            "failed_parameter_extraction": 0,
            "verification_logic_error": 1,
            "incorrect_formula_criteria": 1,
            "computation_error": 0,
            "omitted_calculation_process_or_result": 0,
            "other_error": 0,
            "explanation": (
                "incorrect_formula_criteria: The LLM used the wrong steroid conversion factor. "
                "It chose 1.55 (treating Betamethasone as more potent than Cortisone in a 1:1.55 "
                "ratio), when the correct conversion factor from Cortisone PO to Betamethasone IV "
                "is 0.03 (Cortisone is 25mg-equivalent; Betamethasone is 0.75mg-equivalent per "
                "standard steroid equivalence tables). This produced 404.47 mg instead of the "
                "correct ~7.83 mg. "
                "verification_logic_error: The LLM saw a discrepancy between its own computed value "
                "(404.47 mg) and the claim (7.8299 mg), and immediately labelled the claim 'false' — "
                "without considering that the discrepancy could stem from its own wrong formula. "
                "Since all input parameters are correctly stated, the label should be at least "
                "'partially true'."
            ),
        },
    },

    # ── Type 3: incorrect_formula_criteria | idx=32 ───────────────────────────
    {
        "index": 32,
        "primary_type": "incorrect formula & criteria application",
        "true_label": "partially true",
        "predicted_label": "true",
        "claim": "Based on the patient's data where the Alternative diagnosis to Deep Vein Thrombosis as likely or more likely is yes, the Bedridden recently >3 days is no, the Major surgery within 12 weeks is yes, the Well's DVT Score is -2.",
        "evidence": "A 39-year-old male with a five-year history of moderate ulcerative colitis (UC), as per Montreal classification, was admitted to the emergency department with worsening abdominal and rectal pain secondary to a C. diff infection of one week. Patient had a multi-week history of diarrhea that was waking him up at night leading up to the C. diff diagnosis.",
        "model_reasoning": "The claim is based on the Well's DVT (Deep Vein Thrombosis) Score, which is a clinical prediction rule for assessing the risk of DVT. The score is calculated based on the following criteria: Bedridden recently >3 days (1 point), Major surgery within 12 weeks (1 point), and Alternative diagnosis to DVT as likely or more likely (1 point). The score is then calculated as follows: -2 for each point not present, and +2 for each point present.",
        "annotation": {
            "failed_parameter_extraction": 0,
            "verification_logic_error": 0,
            "incorrect_formula_criteria": 1,
            "computation_error": 0,
            "omitted_calculation_process_or_result": 0,
            "other_error": 0,
            "explanation": (
                "incorrect_formula_criteria: The LLM uses an incorrect Wells DVT scoring rule. "
                "It applies -2 points for each criterion that is not present, which is not part of "
                "the standard Wells DVT scoring system. The correct rule assigns 0 for absent "
                "criteria and +1 for each present criterion."
            ),
        },
    },

    # ── Type 3: incorrect_formula_criteria | idx=23 ───────────────────────────
    {
        "index": 23,
        "primary_type": "incorrect formula & criteria application",
        "true_label": "partially true",
        "predicted_label": "true",
        "claim": "Based on the patient's data where the age is 62 years, the creatinine is 4.1 mg/dL, the sex is Female, the GFR is 12.8919 mL/min/1.73 m².",
        "evidence": "A 62-year-old diabetic woman on chronic hemodialysis for 7 months was admitted to our hospital because of general weakness and constipation for 1 week. She was bed-ridden for 1 month because of a fracture of L3-L4 spine due to a pedestrian traffic accident. Laboratory data showed hemoglobin of 8.6 g/dL, hematocrit 25.8%, random glucose 258 mg/dL, blood urea nitrogen 24.8 mg/dL, serum creatinine 4.1 mg/dL.",
        "model_reasoning": "The MDRD (Modification of Diet in Renal Disease) formula is commonly used to estimate GFR in patients with chronic kidney disease. The formula is: GFR (mL/min/1.73 m2) = 186 x (Serum Creatinine (mg/dL))^-1.154 x (Age (years))^-0.203 x (0.742 if female). Given the patient's age (62 years), sex (female), and creatinine level (4.1 mg/dL), we can plug these values into the formula.",
        "annotation": {
            "failed_parameter_extraction": 0,
            "verification_logic_error": 0,
            "incorrect_formula_criteria": 1,
            "computation_error": 0,
            "omitted_calculation_process_or_result": 0,
            "other_error": 0,
            "explanation": (
                "incorrect_formula_criteria: The LLM uses the MDRD formula with coefficient 186, "
                "but the correct formula uses 175. "
                "LLM's formula: GFR = 186 x creatinine^(-1.154) x age^(-0.203) x (0.742 if female). "
                "Correct formula: GFR = 175 x creatinine^(-1.154) x age^(-0.203) x race_coefficient "
                "x (0.742 if female). "
                "Using 186 instead of 175 produces a higher GFR estimate, leading the LLM to "
                "incorrectly validate the claim as 'true'."
            ),
        },
    },

    # ── Type 4: computation_error | idx=84 ────────────────────────────────────
    {
        "index": 84,
        "primary_type": "computation errors",
        "true_label": "true",
        "predicted_label": "false",
        "claim": "Based on the patient's data where the age is 72 years, the weight is 62.0 kg, the sex is Female, the Sodium is 144.0 mEq/L, the Free Body Water Deficit is 0.7971 L.",
        "evidence": (
            "A 72-year-old woman was admitted to our hospital for surgery for esophageal cancer. "
            "At admission, her blood pressure was 120/70 mmHg and pulse rate was 76 beats per minute. "
            "She was 152 cm in height and weighted 62 Kg. Physical examination was unremarkable. "
            "Preoperatively, renal function was moderately decreased (creatinine, 0.95 mg/dL, "
            "blood urea nitrogen: 16.5 mg/dL, uric acid: 5.7 mg/dL, sodium: 144 mEq/L, "
            "potassium: 4.1 mEq/L, chloride 110 mEq/L, phosphorus 3.7 mg/dL). "
            "During the thoracoscopic esophagectomy, blood loss was 460 ml, and hemodynamics were stable."
        ),
        "model_reasoning": (
            "To determine if the claim is true, we need to calculate the free body water (FBW) and "
            "then the free body water deficit. The formula for calculating FBW is: FBW (L) = "
            "0.58 × weight (kg) for men and 0.49 × weight (kg) for women. Given the patient's "
            "weight is 62 kg and sex is female, we use the female formula. FBW = 0.49 × 62 kg = "
            "30.38 L. The free body water deficit can be calculated using the formula: "
            "Free Body Water Deficit (L) = (0.6 × weight (kg)) / (0.9 × sodium (mEq/L)). "
            "Given the patient's weight is 62 kg and sodium is 144 mEq/L, we can calculate the "
            "free body water deficit. Free Body Water Deficit = (0.6 × 62 kg) / (0.9 × 144 mEq/L) "
            "= 0.6 × 62 kg / 129.6 = 2.34 L. However, the claim states the free body water deficit "
            "is 0.7971 L, which is not consistent with the calculated value."
        ),
        "annotation": {
            "failed_parameter_extraction": 0,
            "verification_logic_error": 1,
            "incorrect_formula_criteria": 1,
            "computation_error": 1,
            "omitted_calculation_process_or_result": 0,
            "other_error": 0,
            "explanation": (
                "computation_error: The LLM applies a wrong formula and also makes an arithmetic "
                "error within it: (0.6 × 62) / (0.9 × 144) = 37.2 / 129.6 ≈ 0.287 L, not 2.34 L "
                "as the LLM states. Its stated result (2.34 L) deviates from the correct value "
                "(0.7971 L) by ~193%, far exceeding the 5% tolerance for equation-based calculators. "
                "incorrect_formula_criteria: The correct formula is TBW% × weight × (Na/140 − 1), "
                "where TBW% = 0.45 for elderly females (age > 64). The LLM instead applies "
                "(0.6 × weight) / (0.9 × sodium), which is not a recognised Free Water Deficit "
                "formula, and uses the adult-female coefficient (0.49) rather than the elderly-female "
                "value (0.45). "
                "verification_logic_error: The LLM skips entity-by-entity verification of input "
                "parameters (age, weight, sex, sodium) against the evidence before calculating, "
                "then labels the claim 'false' despite all inputs being correctly stated."
            ),
        },
    },

    # ── Type 4: computation_error | idx=54 ────────────────────────────────────
    {
        "index": 54,
        "primary_type": "computation errors",
        "true_label": "true",
        "predicted_label": "false",
        "claim": (
            "Based on the patient's data where the creatinine is 2.0 mg/dL, "
            "the Continuous veno-venous hemodialysis for ≥24 hours in the past week is no, "
            "the Dialysis at least twice in the past week is no, "
            "the international normalized ratio is 1.59, the Sodium is 142.0 mEq/L, "
            "the Bilirubin is 0.7 mg/dL, the MELD Na Score is 18.0."
        ),
        "evidence": (
            "A 57-year-old man with a history of hypertension and right hip fracture called "
            "emergency medical services and complained of severe leg pain. "
            "Initial investigations showed blood urea nitrogen 29 mg/dL, creatinine 2.0 mg/dL, "
            "sodium 142 mEq/L, potassium 3.9 mEq/L, lactic acid 4.25 mmol/L. "
            "The liver function test demonstrated total bilirubin 0.7 mg/dL, "
            "total protein 6.2 gm/dL, and albumin 3.5 mg/dL. "
            "Coagulogram showed prothrombin time 17.5 seconds, partial thromboplastin time "
            "25.8 seconds, and international normalized ratio 1.59. "
            "No dialysis was performed during admission."
        ),
        "model_reasoning": (
            "The MELD Na Score is calculated using the formula: "
            "4.43 * log(e)(bilirubin mg/dL) + 10.36 * log(e)(INR) + 3.78 * log(e)(Na mEq/L) + 1.18. "
            "Given the values: bilirubin = 0.7 mg/dL, INR = 1.59, and Na = 142 mEq/L, "
            "we can calculate the MELD Na Score as follows: "
            "4.43 * log(e)(0.7) + 10.36 * log(e)(1.59) + 3.78 * log(e)(142) + 1.18. "
            "This calculation results in a MELD Na Score of approximately 9.3, not 18.0. "
            "Therefore, the claim that the MELD Na Score is 18.0 is false."
        ),
        "annotation": {
            "failed_parameter_extraction": 1,
            "verification_logic_error": 1,
            "incorrect_formula_criteria": 1,
            "computation_error": 1,
            "omitted_calculation_process_or_result": 0,
            "other_error": 0,
            "explanation": (
                "computation_error: The LLM's own formula evaluates to ~23.14, not 9.3 — an "
                "additional arithmetic error. Its stated result (9.3) deviates from the correct "
                "value (18.0) by ~48%, far exceeding the 5% tolerance for equation-based calculators. "
                "incorrect_formula_criteria: The LLM applies an incorrect MELD Na formula that omits "
                "creatinine entirely. The correct two-step UNOS/OPTN formula is: "
                "MELD(i) = 0.957 × ln(Cr) + 0.378 × ln(bilirubin) + 1.120 × ln(INR) + 0.643; "
                "if MELD(i) × 10 > 11, then MELD = MELD(i) + 1.32 × (137 − Na) − "
                "[0.033 × MELD(i) × (137 − Na)], with Na clamped to [125, 137] and bilirubin "
                "floored at 1.0. "
                "failed_parameter_extraction: The LLM omits creatinine from its formula and ignores "
                "the two dialysis conditions (CVVHD ≥24 h and dialysis ≥2× per week), which are "
                "required inputs for the MELD Na calculation. "
                "verification_logic_error: The LLM skips entity verification and directly computes "
                "the score, labelling the claim 'false' despite all input entities being correctly "
                "stated — the correct label would be 'partially true' if the computation were wrong."
            ),
        },
    },

    # ── Type 5: omitted_calculation_process_or_result | idx=2 ─────────────────
    {
        "index": 2,
        "primary_type": "omitting the calculation process or result",
        "true_label": "partially true",
        "predicted_label": "true",
        "claim": "Based on the patient's data where the sex is Male, the age is 60 years, the weight is 58.5 kg, the height is 163.1 cm, the creatinine is 10.7 mg/dL, the Creatinine Clearance is 5.225 mL/min.",
        "evidence": "A 60-year-old male with end-stage kidney disease due to ADPKD was admitted to our hospital for an evaluation of diplopia and left hemiplegia in March 2012. He had started maintenance hemodialysis in 2005. He had hypertension, which was relatively under good control with antihypertensive agents.",
        "model_reasoning": "The claim provides the patient's data, including sex, age, weight, height, creatinine level, and estimated creatinine clearance. The patient's sex and age match the provided information. The weight and height are also consistent with the data. The creatinine level is 10.7 mg/dL, which is consistent with the data. The estimated creatinine clearance is 5.225 mL/min, which is calculated based on the patient's creatinine level and age. This calculation is consistent with the patient's end-stage kidney disease due to ADPKD. Therefore, the claim is accurate based on the provided evidence.",
        "annotation": {
            "failed_parameter_extraction": 0,
            "verification_logic_error": 0,
            "incorrect_formula_criteria": 0,
            "computation_error": 0,
            "omitted_calculation_process_or_result": 1,
            "other_error": 0,
            "explanation": (
                "omitted_calculation_process_or_result: The LLM did not apply the Cockcroft-Gault "
                "formula to compute Creatinine Clearance. Instead, it verified that the listed input "
                "parameters (sex, age, weight, height, creatinine) were consistent with the evidence, "
                "then accepted the claim's stated CrCl value (5.225 mL/min) as accurate without "
                "performing the calculation."
            ),
        },
    },

    # ── Type 6: other_error | idx=14 ──────────────────────────────────────────
    {
        "index": 14,
        "primary_type": "other errors",
        "true_label": "partially true",
        "predicted_label": "unknown",
        "claim": "Based on the patient's data where the weight is 77.0 kg, the height is 183.0 cm, the BMI is 25.9232 kg/m2.",
        "evidence": "An 81-year-old, 77 kg, 183 cm man with a three-year history of muscle biopsy-proven IBM suffered a left intertrochanteric femoral fracture and a small parafalcine subdural hematoma as a result of falling down stairs in his home.",
        "model_reasoning": "To calculate the patient's BMI, we need to divide the weight in kilograms by the square of the height in meters. The patient's weight is 77 kg and height is 183 cm. First, we convert the height from centimeters to meters: 183 cm = 1.83 m. Then, we calculate the square of the height: 1.83 m^2 = 3.3489 m^2. Next, we divide the weight by the square of the height: 77 kg / 3.3489 m^2 = 23.0 kg/m^2. Rounding to four decimal places as in the claim, we get 23.0 kg/m^2 is incorrect, the correct BMI is 23.0 is incorrect, the correct BMI is 23.0 is incorrect, the correct BMI is 23.0 is",
        "annotation": {
            "failed_parameter_extraction": 0,
            "verification_logic_error": 0,
            "incorrect_formula_criteria": 0,
            "computation_error": 0,
            "omitted_calculation_process_or_result": 0,
            "other_error": 1,
            "explanation": (
                "other_error: The LLM entered an infinite reasoning loop, repeatedly producing "
                "'the correct BMI is 23.0 is incorrect' without converging, and output 'UNKNOWN' "
                "as its label. This is a model generation failure (token repetition / degenerate "
                "output) that does not fit any of the five standard error categories. The formula "
                "and input parameters are correct; the failure is entirely in the generation process."
            ),
        },
    },
]
