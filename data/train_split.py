import csv
import random
from collections import defaultdict

random.seed(42)

with open('medcalc_train_claim_full.csv', newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    headers = reader.fieldnames

# Group by (Calculator ID, Label)
groups = defaultdict(list)
for r in rows:
    groups[(r['Calculator ID'], r['Label'])].append(r)

labels = ['false', 'partially true', 'true']
calc_ids = sorted(set(r['Calculator ID'] for r in rows), key=lambda x: int(x))

from collections import Counter

def write_csv(path, data):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)

def verify(name, data):
    label_dist = Counter(r['Label'] for r in data)
    calc_dist = Counter(r['Calculator ID'] for r in data)
    print(f"{name}: {len(data)} rows | labels: {dict(label_dist)} | unique calcs: {len(calc_dist)}")

# --- File 1: 1 per (calc, label) = 165 rows ---
file1_rows = []
for cid in calc_ids:
    for label in labels:
        file1_rows.append(random.choice(groups[(cid, label)]))

write_csv('sample_set_1.csv', file1_rows)
verify('sample_set_1.csv', file1_rows)

# --- File 2: 2 per (calc, label) = 330 rows, non-overlapping with file1 ---
used1 = set(id(r) for r in file1_rows)
file2_rows = []
for cid in calc_ids:
    for label in labels:
        pool = [r for r in groups[(cid, label)] if id(r) not in used1]
        file2_rows.extend(random.sample(pool, 2))

write_csv('sample_set_2.csv', file2_rows)
verify('sample_set_2.csv', file2_rows)

# --- File 3: 7 per (calc, label) = 1155 rows, non-overlapping with file1 & file2 ---
used = set(id(r) for r in file1_rows + file2_rows)
file3_rows = []
for cid in calc_ids:
    for label in labels:
        pool = [r for r in groups[(cid, label)] if id(r) not in used]
        file3_rows.extend(random.sample(pool, 7))

write_csv('sample_set_3.csv', file3_rows)
verify('sample_set_3.csv', file3_rows)
