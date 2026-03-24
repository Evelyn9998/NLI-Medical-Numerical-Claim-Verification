import csv
import random
from collections import defaultdict, Counter

random.seed(42)

with open('../mrt_claim_full.csv', newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    headers = reader.fieldnames

groups = defaultdict(list)
for r in rows:
    groups[(r['Calculator ID'], r['Label'])].append(r)

labels = ['false', 'partially true', 'true']
calc_ids = sorted(set(r['Calculator ID'] for r in rows), key=lambda x: int(x))


def write_csv(path, data):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)


def verify(name, data):
    label_dist = Counter(r['Label'] for r in data)
    calc_dist = Counter(r['Calculator ID'] for r in data)
    print(f"{name}: {len(data)} rows | labels: {dict(label_dist)} | unique calcs: {len(calc_dist)}")


def sample_up_to(pool, k):
    return random.sample(pool, min(k, len(pool)))


def sample_prefer_nonoverlap(full_pool, used, k):
    """Try to sample k items from full_pool preferring non-used ones.
    If non-overlapping pool is insufficient, fall back to the full pool."""
    fresh = [r for r in full_pool if id(r) not in used]
    if len(fresh) >= k:
        return random.sample(fresh, k)
    return sample_up_to(full_pool, k)


# --- File 1: up to 1 per (calc, label) ---
file1_rows = []
for cid in calc_ids:
    for label in labels:
        file1_rows.extend(sample_up_to(groups[(cid, label)], 1))

write_csv('mrt_train_1per.csv', file1_rows)
verify('mrt_train_1per.csv', file1_rows)

# --- File 2: up to 2 per (calc, label), non-overlapping with File 1 (fallback: allow overlap) ---
used1 = set(id(r) for r in file1_rows)
file2_rows = []
for cid in calc_ids:
    for label in labels:
        file2_rows.extend(sample_prefer_nonoverlap(groups[(cid, label)], used1, 2))

write_csv('mrt_train_2per.csv', file2_rows)
verify('mrt_train_2per.csv', file2_rows)

# --- File 3: up to 7 per (calc, label), non-overlapping with File 1 & 2 (fallback: allow overlap) ---
used12 = set(id(r) for r in file1_rows + file2_rows)
file3_rows = []
for cid in calc_ids:
    for label in labels:
        file3_rows.extend(sample_prefer_nonoverlap(groups[(cid, label)], used12, 7))

write_csv('mrt_train_7per.csv', file3_rows)
verify('mrt_train_7per.csv', file3_rows)

# --- Report calculators with insufficient samples ---
print("\nCalculators with fewer than 7 per label:")
for cid in calc_ids:
    for label in labels:
        cnt = len(groups[(cid, label)])
        if cnt < 7:
            name = groups[(cid, label)][0]['Calculator Name']
            print(f"  calc {cid} ({name}) / {label}: {cnt} total samples")
