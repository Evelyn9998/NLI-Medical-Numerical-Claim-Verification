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
Type 4 – computation_error                    : idx 123, 89
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
                "date difference happened to match the gestational age claim. The LLM failed to follow "
                "the correct labelling logic: even though the parameter extraction is wrong, it still "
                "wrongly verdicted the claim as 'partially true'."
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
                "'partially true'. The LLM failed to follow the logic of 'partially true' in the "
                "prompt — it did not extract and verify input parameters first before issuing a verdict."
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
                "evidence. The LLM made a qualitative judgment based on 'likely' risk rather than "
                "applying the Revised Cardiac Risk Index (RCRI) formula, and labelled the claim "
                "'false' without performing the actual score calculation. The LLM lacked knowledge "
                "of RCRI criteria and failed to follow the 'partially true' logic in the prompt — "
                "it did not extract and verify each parameter first before issuing a verdict."
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
                "verification_logic_error: Because the LLM's computed value (404.47 mg) was vastly "
                "different from the claim (7.83 mg), it labelled the claim 'false' without "
                "recognising that all input parameters were correctly extracted."
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
                "criteria and +1 for each present criterion (with -2 only for 'alternative diagnosis "
                "as likely or more likely'). The LLM also wrongly subtracts points for non-existent "
                "entities. Using the correct criteria, the score should be +1 (major surgery within "
                "12 weeks: yes) - 2 (alternative diagnosis likely: yes) = -1, not -2."
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

    # ── Type 4: computation_error | idx=123 ───────────────────────────────────
    {
        "index": 123,
        "primary_type": "computation errors",
        "true_label": "true",
        "predicted_label": "partially true",
        "claim": "Based on the patient's data where the Heart Rate or Pulse is 139 beats per minute, the QT Interval is 468 msec, the Corrected QT Interval is 673.4 msec.",
        "evidence": "A patient has a heart rate of 139 bpm and a QT interval of 468 msec.",
        "model_reasoning": "To determine the correctness of the claim, we need to calculate the Corrected QT Interval (QTc) using the Bazett's formula: QTc = QT / sqrt(RR), where RR is the interval between heartbeats in seconds. Given the heart rate of 139 bpm, the RR interval is 1/139 seconds. First, convert the RR interval to seconds: 1/139 approximately 0.00718 seconds. Then, calculate the QTc: QTc = 468 / sqrt(0.00718) approximately 468 / 0.084 approximately 5571.43 / 8.3 approximately 671.7 msec. However, the claim states the QTc is 673.4 msec, which is close but not exactly equal to our calculated value.",
        "annotation": {
            "failed_parameter_extraction": 0,
            "verification_logic_error": 0,
            "incorrect_formula_criteria": 1,
            "computation_error": 1,
            "omitted_calculation_process_or_result": 0,
            "other_error": 0,
            "explanation": (
                "computation_error: The LLM makes multiple arithmetic errors. "
                "1/139 = 0.00719, not 0.00718. "
                "5571.43 / 8.3 = 671.25662651, not 671.7. "
                "These errors produce a result that diverges from the correct value (673.4 msec), "
                "causing the LLM to incorrectly label the claim as 'partially true'. "
                "incorrect_formula_criteria: The LLM selects Bazett's formula (QTc = QT / sqrt(RR)) "
                "instead of the formula used by the dataset to derive the ground-truth value of "
                "673.4 msec. Different QTc formulas (Bazett, Fridericia, Framingham) produce "
                "different results for the same inputs."
            ),
        },
    },

    # ── Type 4: computation_error | idx=89 ────────────────────────────────────
    {
        "index": 89,
        "primary_type": "computation errors",
        "true_label": "partially true",
        "predicted_label": "true",
        "claim": "Based on the patient's data where the creatinine is 0.25 mg/dL, the Urine sodium is 16.0 mmol/L, the Urine creatinine is 6.8 mg/dL, the Sodium is 123.0 mmol/L, the FEna Percentage is 0.4165 %.",
        "evidence": "A 50-year-old male visited our emergency room(ER) because of generalized edema. He appeared remarkably swollen during the past week, with associated reduced urine output. Laboratory data showed serum creatinine 0.25 mg/dL. Serum sodium was 123.0 mmol/L. Urine sodium was 16.0 mmol/L. Urine creatinine was 6.8 mg/dL.",
        "model_reasoning": "To determine the validity of the claim, we need to calculate the fractional excretion of sodium (FEna) using the formula: FEna = (Urine sodium / Serum sodium) / (Urine creatinine / Serum creatinine) * 100. Given the values, FEna = (16.0 / 123.0) / (6.8 / 0.25) * 100 = 0.4165%. This calculation matches the provided FEna percentage, indicating that the claim is accurate.",
        "annotation": {
            "failed_parameter_extraction": 0,
            "verification_logic_error": 0,
            "incorrect_formula_criteria": 0,
            "computation_error": 1,
            "omitted_calculation_process_or_result": 0,
            "other_error": 0,
            "explanation": (
                "computation_error: The LLM's formula is correct "
                "(FEna = (UNa / SNa) / (UCr / SCr) x 100), and all input values are correctly "
                "extracted, but the arithmetic is wrong. "
                "Correct calculation: (16.0 / 123.0) / (6.8 / 0.25) x 100 = 0.4782%, not 0.4165%. "
                "The LLM incorrectly confirmed 0.4165% as accurate and labelled the claim 'true', "
                "when the correct label is 'partially true' because the claimed FEna value is wrong."
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
                "performing the calculation. The correct calculation yields a different value, making "
                "the claim 'partially true', not 'true'."
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
