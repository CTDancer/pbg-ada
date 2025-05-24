import pandas as pd
import os

folder_path = '/home/tc415/PPI_datasets'

dataframes = []

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        # Keep only the specified columns
        df = df[['Binder', 'Target']]
        # Append the DataFrame to the list
        dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
print(f"Total: {len(combined_df)}")

combined_df = combined_df.drop_duplicates()
print(f"Drop Duplicates: {len(combined_df)}")

combined_df = combined_df[combined_df['Binder'].str.len() + combined_df['Target'].str.len() <= 1024]
print(f"Drop Length over 1024: {len(combined_df)}")

# Optionally, save the resulting DataFrame to a new CSV file
output_path = '/home/tc415/PPI_datasets/ppiref_filtered.csv'
combined_df.to_csv(output_path, index=False)

print(f"Filtered data saved to {output_path}")
