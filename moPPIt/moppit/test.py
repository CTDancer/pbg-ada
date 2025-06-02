import pandas as pd

# Load the CSV files
df = pd.read_csv('/home/tc415/moPPIt/dataset/pep_prot/correct_pepnn_biolip.csv')
a = pd.read_csv('/home/tc415/moPPIt/moppit/test.csv')

# Initialize lists to store binders and targets
targets = []
binders = []

# Extract targets and binders from the 'Sequence' column of 'a'
for i in range(len(a['Sequence'])):
    if 'raw' in a['ID'][i]:
        sequence = a['Sequence'][i].split(':')
        targets.append(sequence[0])
        binders.append(sequence[1])

# Iterate over the extracted targets and binders to find matching rows in df
for i in range(len(targets)):
    binder = binders[i]
    target = targets[i]
    matching_indices = df[(df['Binder'] == binder) & (df['Target'] == target)].index.tolist()
    if matching_indices:
        print(matching_indices[0])
    else:
        print(f"No match found for Binder: {binder}, Target: {target}")
