import pandas as pd

def replace_label(label):
    if 'Phosphotyrosine'.lower() in label.lower():
        return 'Phosphotyrosine'
    elif 'Phosphoserine'.lower() in label.lower():
        return 'Phosphoserine'
    elif 'Phosphothreonine'.lower() in label.lower():
        return 'Phosphothreonine'
    elif 'N6-acetyllysine'.lower() in label.lower():
        return 'N6-acetyllysine'
    else:
        return label





labels_to_keep = [
    "N-linked (GlcNAc...) asparagine",
    "Pyrrolidone carboxylic acid",
    "Phosphoserine",
    "Phosphothreonine",
    "N-acetylalanine",
    "N-acetylmethionine",
    "N6-acetyllysine",
    "Phosphotyrosine",
    "S-diacylglycerol cysteine",
    "N6-(pyridoxal phosphate)lysine",
    "N-acetylserine",
    "N6-carboxylysine",
    "N6-succinyllysine",
    "S-palmitoyl cysteine",
    "O-(pantetheine 4'-phosphoryl)serine",
    "Phosphotyrosine; by autocatalysis",
    "Sulfotyrosine",
    "O-linked (GalNAc...) threonine",
    "Omega-N-methylarginine",
    "N-myristoyl glycine",
    "4-hydroxyproline",
    "Asymmetric dimethylarginine",
    "N5-methylglutamine",
    "4-aspartylphosphate",
    "S-geranylgeranyl cysteine",
    "4-carboxyglutamate",
]


df = pd.read_csv('sprot_labels.csv',index_col=[0])
df.dropna(inplace=True)
df.drop(df.filter(regex="Unnamed"),axis=1, inplace=True)
df['label'] = df['label'].map(replace_label)

df['pos'] = (df['pos'].astype(int) - 1).astype(int)
df['ori_seq'] = df['ori_seq'].str.upper()
df['token'] = '<' + df['label'] + '>'
df = df[df['label'].isin(labels_to_keep)]
df.to_csv('ptm_labels.csv')