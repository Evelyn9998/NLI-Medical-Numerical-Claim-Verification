import pandas as pd

df = pd.read_csv('test_data_mr.csv')

cids_to_remove = [13, 28, 11, 36]
df = df[~df['Calculator ID'].isin(cids_to_remove)]


rows_to_drop = [
    451, 472, 803, 473, 946, 940, 804, 764, 936, 761, 798, 930, 738, 934, 938, 789, 469,
    792, 948, 944, 937, 781, 941, 507, 478, 801, 945, 931, 477, 806, 929, 763, 794,
    471, 932, 481, 942, 947, 943, 805, 486, 939, 768, 810, 933, 935, 468
]

df = df.drop(index=rows_to_drop, errors='ignore')


df.to_csv('test_data_mr_cleaned.csv', index=False)