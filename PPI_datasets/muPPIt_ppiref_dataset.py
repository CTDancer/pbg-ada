import pandas as pd

input_csv_path = '/home/tc415/PPI_datasets/ppiref_filtered.csv'

# Read the filtered CSV file
filtered_df = pd.read_csv(input_csv_path)

# Create 'Sequence' and 'Length' columns
filtered_df['Sequence'] = filtered_df['Target'] + '-' + filtered_df['Binder']
filtered_df['Length'] = filtered_df['Sequence'].str.len()

# Select only the 'Sequence' and 'Length' columns for the output
output_df = filtered_df[['Sequence', 'Length']]
print(len(output_df))

output_df = output_df[output_df['Length'] >= 25]
print(len(output_df))

# Define the output path for the new CSV file
output_csv_path = '/home/tc415/PPI_datasets/muPPIt_ppiref_dataset.csv'

# Save the resulting DataFrame to a new CSV file
output_df.to_csv(output_csv_path, index=False)

print(f"Sequences and lengths saved to {output_csv_path}")
