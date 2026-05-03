"""
Annotation Agreement Analysis
==============================
Finds samples where both annotators (WT and Evelyn) agree on each error type,
tags "pure" samples (only 1 agreed error type — best for few-shot),
and prints a summary with annotation snippets from both annotators.

Input files:
  error_analysis_50_wt_1.xlsx      — WT binary labels
  error_analysis_50_Evelyn_1.xlsx  — Evelyn binary labels

Output:
  annotation_agreement_results.txt — full results saved to file
  (also printed to stdout)

Usage:
  python annotation_agreement_analysis.py
"""

import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────────

ERROR_COLS = [
    'failed parameter extraction',
    'verification logic errors',
    'incorrect formula & criteria application',
    'computation errors',
    'omitting the calculation process or result',
    'other errors',
]

FEW_SHOT_INDICES   = {77, 38, 151, 11, 36, 57, 32, 23, 123, 89, 2, 14}
VALIDATION_INDICES = {18, 52, 95, 50, 35, 53, 74, 86, 9}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load labels
    df_wt = pd.read_excel('error_analysis_50_wt_1.xlsx').set_index('index')
    df_ev = pd.read_excel('error_analysis_50_Evelyn_1.xlsx').set_index('index')

    # Find agreed-positive samples per error type
    def get_agreed_types(idx):
        return [col for col in ERROR_COLS
                if (int(df_wt.loc[idx, col]) == 1
                    and int(df_ev.loc[idx, col]) == 1)]

    agreed = {col: [] for col in ERROR_COLS}
    for idx in df_wt.index:
        if idx not in df_ev.index:
            continue
        for col in ERROR_COLS:
            try:
                if int(df_wt.loc[idx, col]) == 1 and int(df_ev.loc[idx, col]) == 1:
                    agreed[col].append(int(idx))
            except (ValueError, KeyError):
                pass

    # Build output lines
    lines = []
    lines.append('=' * 72)
    lines.append('ANNOTATION AGREEMENT ANALYSIS — AGREED-POSITIVE SAMPLES')
    lines.append('★ = pure sample (only 1 agreed error type — best for few-shot)')
    lines.append('=' * 72)
    lines.append('')

    for col in ERROR_COLS:
        idxs = agreed[col]
        lines.append(f'■ {col}  (n={len(idxs)})')
        for idx in idxs:
            types   = get_agreed_types(float(idx))
            n_types = len(types)
            star    = '★' if n_types == 1 else f'  [{n_types} types]'
            tl = df_wt.loc[float(idx), 'true_label']
            pl = df_wt.loc[float(idx), 'predicted_label']

            # Role tags
            role = ''
            if idx in FEW_SHOT_INDICES:
                role = '  [FEW-SHOT]'
            elif idx in VALIDATION_INDICES:
                role = '  [VALIDATION]'

            lines.append(f'   {star} idx={idx:<4}  true={str(tl):<16} pred={str(pl):<16}{role}')
        lines.append('')

    # Few-shot / validation summary
    lines.append('=' * 72)
    lines.append('RECOMMENDED FEW-SHOT & VALIDATION SPLIT')
    lines.append('=' * 72)
    rows = [
        ('failed parameter extraction',               'idx=77, 38',      'idx=18, 52'),
        ('verification logic errors',                 'idx=151, 11, 36', 'idx=95, 50'),
        ('incorrect formula & criteria application',  'idx=57, 32, 23',  'idx=35'),
        ('computation errors',                        'idx=123, 89',     'idx=53'),
        ('omitting the calculation process or result','idx=2',           'idx=74, 86, 9'),
        ('other errors',                              'idx=14',          '(none)'),
    ]
    lines.append(f'  {"Error Type":<46}  {"Few-shot":<20}  Validation')
    lines.append('  ' + '-' * 80)
    for et, fs, val in rows:
        lines.append(f'  {et:<46}  {fs:<20}  {val}')
    lines.append('')
    lines.append(f'  Total few-shot  : {len(FEW_SHOT_INDICES)} samples  '
                 f'(indices: {sorted(FEW_SHOT_INDICES)})')
    lines.append(f'  Total validation: {len(VALIDATION_INDICES)} samples  '
                 f'(indices: {sorted(VALIDATION_INDICES)})')
    lines.append('')

    output = '\n'.join(lines)
    print(output)

    with open('annotation_agreement_results.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print('Saved: annotation_agreement_results.txt')


if __name__ == '__main__':
    main()
