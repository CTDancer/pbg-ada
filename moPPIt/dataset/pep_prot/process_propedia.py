import pandas as pd

df = pd.read_csv('/home/tc415/moPPIt/dataset/pep_prot/complex_propedia_with_binding_residues.csv')

new_df = df[['Peptide Sequence', 'Receptor Sequence', 'binding_residues_receptor']]

new_df = new_df.rename(columns={'Peptide Sequence': 'Binder'})
new_df = new_df.rename(columns={'Receptor Sequence': 'Target'})
new_df = new_df.rename(columns={'binding_residues_receptor': 'Motif'})

new_df_filtered = new_df[~new_df['Binder'].str.contains('X')]
new_df_filtered = new_df_filtered[~new_df_filtered['Target'].str.contains('X')]
new_df_filtered['Motif'] = new_df_filtered['Motif'].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

new_df_filtered.to_csv('/home/tc415/moPPIt/dataset/pep_prot/complex_propedia_filtered.csv', index=False)

