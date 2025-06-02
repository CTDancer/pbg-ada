import pandas as pd
import pdb

string = ''
with open('/home/tc415/moPPIt/moppit/human_e3_4.out', 'r') as f:
    lines = f.readlines()

for line in lines:
    string += line.strip()
    string += '\n'

df = pd.DataFrame(columns=['Entry', 'Binders (1~10)', 'Binders (11~20)', 'Binding sites'])

results = string.split('################ Processing Entry ')

data = []

for result in results:
    entry = result.split(' ')[0]

    ls = result.split('\n')
    binding_site = None
    for l in ls:
        if 'Candidate Binding Site = ' in l:
            binding_site = l.split('Candidate Binding Site = ')[-1].strip()
    
    if binding_site == None:
        data.append({'Entry':entry, 'Binders (1~10)':None, 'Binders (11~20)':None, 'Binding sites':None})
        continue
    
    # pdb.set_trace()
    final_binders = result.split('$$$$$$$$$$\n')[-1]
    binders = final_binders.split('\n')[:20]
    binders_1 = ''
    binders_2 = ''
    for binder in binders[:10]:
        binders_1 += binder
        binders_1 += '\n'
    for binder in binders[10:20]:
        binders_2 += binder
        binders_2 += '\n'
    
    # pdb.set_trace()

    data.append({'Entry':entry, 'Binders (1~10)':binders_1.strip(), 'Binders (11~20)':binders_2.strip(), 'Binding sites':binding_site})

data_df = pd.DataFrame(data)
data_df.to_csv('/home/tc415/moPPIt/moppit/human_e3_4.csv', index=False)
data_df.to_excel('/home/tc415/moPPIt/moppit/human_e3_4.xlsx', index=False)