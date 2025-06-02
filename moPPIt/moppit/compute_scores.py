# Local Interaction Score calculation
# It's designed for ColabFold-derived outputs (json and pdb files)

###################################################################################################################
### CHANGE THIS BEFORE RUNNING ONLY
base_path = "/home/tc415/moPPIt/moppit/pdb_results/EWSFLI/GPSSWYS" # where you want to pull files from 
clear_old_outputs = True # get rid of pae_12_analysis files before starting, instead of adding on
###################################################################################################################

import os
import json
import statistics
import numpy as np
from Bio import PDB
import pandas as pd
from multiprocessing import Pool
from pandas.errors import EmptyDataError
import subprocess
import time
import pdb

def calculate_pae(pdb_file_path: str, print_results: bool = True, pae_cutoff: float = 12.0, name_separator: str = "___"):
    '''
    Calculates PAE on an input pdb file, located at pdb_file_path
    '''
    parser = PDB.PDBParser()
    file_name = pdb_file_path.split("/")[-1]
    data_folder = pdb_file_path.split("/")[-2]

    if 'rank' not in file_name:
        if print_results:
            print(f"Skipping {file_name} as it does not contain 'rank' in the file name.")
        return None

    # Splitting the file name first: "_relaxed" if analyzing relaxed files, "_unrelaxed" if analyzing unrelaxed files
    relaxed = True
    relax_split = "_relaxed"
    if "_relaxed" not in file_name:
        relaxed = False
        relax_split = "_unrelaxed"
    
    parts = file_name.split(relax_split)
    if len(parts) < 2:
        if print_results:
            print(f"Warning: File {file_name} does not follow expected {relax_split} naming convention. Skipping this file.")
        return None

    # Using name_separator to separate protein_1 and protein_2 from the first part
    ### Sophie 
    ID = file_name.split(relax_split)[0]
    pae_file_name = data_folder + '+' + ID + '_pae.png'
    #pae_file_name = data_folder + '+' + protein_1 + name_separator + protein_2 + '_pae.png'

    # Extract rank information from protein_2_temp
    if f"{relax_split}_rank_00" in file_name:
        rank_temp = file_name.split(f"{relax_split}_rank_00")[1]
        rank = rank_temp.split("_alphafold2")[0]
    else:
        rank = "Not Available"  # or any default value you prefer

    if print_results:
        print("Rank:", rank)
    
    repl = relax_split.split("_")[1]
    json_file = pdb_file_path.replace(".pdb", ".json").replace(f"{repl}", "scores")
    structure = parser.get_structure("example", pdb_file_path)

    protein_a_len = 0
    protein_b_len = 0
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            chain_length = sum(1 for _ in chain.get_residues())
            if chain_id == 'A':
                protein_a_len = chain_length
            else:
                protein_b_len = chain_length
                
            if print_results:
                print(f"Chain {chain_id} length : {chain_length}")


    # Load the JSON file
    with open(json_file, 'r') as file:
        json_data = json.load(file)
    
    plddt = statistics.mean(json_data["plddt"])
    ptm = json_data["ptm"]
    iptm = json_data["iptm"]
    pae = np.array(json_data['pae'])

    # Calculate thresholded_pae
    thresholded_pae = np.where(pae < pae_cutoff, 1, 0)

    # Calculate the interaction amino acid numbers
    local_interaction_protein_a = np.count_nonzero(thresholded_pae[:protein_a_len, :protein_a_len])
    local_interaction_protein_b = np.count_nonzero(thresholded_pae[protein_a_len:, protein_a_len:])
    local_interaction_interface_1 = np.count_nonzero(thresholded_pae[:protein_a_len, protein_a_len:])
    local_interaction_interface_2 = np.count_nonzero(thresholded_pae[protein_a_len:, :protein_a_len])
    local_interaction_interface_avg = (
        local_interaction_interface_1 + local_interaction_interface_2
    )

    
    # Calculate average thresholded_pae for each region
    average_thresholded_protein_a = thresholded_pae[:protein_a_len,:protein_a_len].mean() * 100
    average_thresholded_protein_b = thresholded_pae[protein_a_len:,protein_a_len:].mean() * 100
    average_thresholded_interaction1 = thresholded_pae[:protein_a_len,protein_a_len:].mean() * 100
    average_thresholded_interaction2 = thresholded_pae[protein_a_len:,:protein_a_len].mean() * 100
    average_thresholded_interaction_total = (average_thresholded_interaction1 + average_thresholded_interaction2) / 2
    

    pae_protein_a = np.mean( pae[:protein_a_len,:protein_a_len] )
    pae_protein_b = np.mean( pae[protein_a_len:,protein_a_len:] )
    pae_interaction1 = np.mean(pae[:protein_a_len,protein_a_len:])
    pae_interaction2 = np.mean(pae[protein_a_len:,:protein_a_len])
    pae_interaction_total = ( pae_interaction1 + pae_interaction2 ) / 2

    # For pae_A
    selected_values_protein_a = pae[:protein_a_len, :protein_a_len][thresholded_pae[:protein_a_len, :protein_a_len] == 1]
    average_selected_protein_a = np.mean(selected_values_protein_a)

    # For pae_B
    selected_values_protein_b = pae[protein_a_len:, protein_a_len:][thresholded_pae[protein_a_len:, protein_a_len:] == 1]
    average_selected_protein_b = np.mean(selected_values_protein_b)

    # For pae_interaction1
    selected_values_interaction1 = pae[:protein_a_len, protein_a_len:][thresholded_pae[:protein_a_len, protein_a_len:] == 1]
    average_selected_interaction1 = np.mean(selected_values_interaction1) if selected_values_interaction1.size > 0 else pae_cutoff

    # For pae_interaction2
    selected_values_interaction2 = pae[protein_a_len:, :protein_a_len][thresholded_pae[protein_a_len:, :protein_a_len] == 1]
    average_selected_interaction2 = np.mean(selected_values_interaction2) if selected_values_interaction2.size > 0 else pae_cutoff

    # For pae_interaction_total
    average_selected_interaction_total = (average_selected_interaction1 + average_selected_interaction2) / 2

    if print_results:
        # Print the total results
        print("Total pae_A : {:.2f}".format(pae_protein_a))
        print("Total pae_B : {:.2f}".format(pae_protein_b))
        print("Total pae_i_1 : {:.2f}".format(pae_interaction1))
        print("Total pae_i_2 : {:.2f}".format(pae_interaction2))
        print("Total pae_i_avg : {:.2f}".format(pae_interaction_total))

        # Print the local results
        print("Local pae_A : {:.2f}".format(average_selected_protein_a))
        print("Local pae_B : {:.2f}".format(average_selected_protein_b))
        print("Local pae_i_1 : {:.2f}".format(average_selected_interaction1))
        print("Local pae_i_2 : {:.2f}".format(average_selected_interaction2))
        print("Local pae_i_avg : {:.2f}".format(average_selected_interaction_total))

        # Print the >PAE-cutoff area
        print("Local interaction area (Protein A):", local_interaction_protein_a)
        print("Local interaction area (Protein B):", local_interaction_protein_b)
        print("Local interaction area (Interaction 1):", local_interaction_interface_1)
        print("Local interaction area (Interaction 2):", local_interaction_interface_2)
        print("Total Interaction area (Interface):", local_interaction_interface_avg)


    # Transform the pae matrix
    scaled_pae = reverse_and_scale_matrix(pae, pae_cutoff)

    # For local interaction score for protein_a
    selected_values_protein_a = scaled_pae[:protein_a_len, :protein_a_len][thresholded_pae[:protein_a_len, :protein_a_len] == 1]
    average_selected_protein_a_score = np.mean(selected_values_protein_a)

    # For local interaction score for protein_b
    selected_values_protein_b = scaled_pae[protein_a_len:, protein_a_len:][thresholded_pae[protein_a_len:, protein_a_len:] == 1]
    average_selected_protein_b_score = np.mean(selected_values_protein_b)

    # For local interaction score1
    selected_values_interaction1_score = scaled_pae[:protein_a_len, protein_a_len:][thresholded_pae[:protein_a_len, protein_a_len:] == 1]
    average_selected_interaction1_score = np.mean(selected_values_interaction1_score) if selected_values_interaction1_score.size > 0 else 0

    # For local interaction score2
    selected_values_interaction2_score = scaled_pae[protein_a_len:, :protein_a_len][thresholded_pae[protein_a_len:, :protein_a_len] == 1]
    average_selected_interaction2_score = np.mean(selected_values_interaction2_score) if selected_values_interaction2_score.size > 0 else 0

    # For average local interaction score
    average_selected_interaction_total_score = (average_selected_interaction1_score + average_selected_interaction2_score) / 2
    
    if print_results:
        # Print the local interaction scores
        print("Local Interaction Score_A : {:.3f}".format(average_selected_protein_a_score))
        print("Local Interaction Score_B : {:.3f}".format(average_selected_protein_b_score))
        print("Local Interaction Score_i_1 : {:.3f}".format(average_selected_interaction1_score))
        print("Local Interaction Score_i_2 : {:.3f}".format(average_selected_interaction2_score))
        print("Local Interaction Score_i_avg : {:.3f}".format(average_selected_interaction_total_score))

    COLUMNS_ORDER = [
        'ID', 'pLDDT', 'pTM', 'ipTM',
        'Local_Score_A', 'Local_Score_B', 'Local_Score_i_1', 'Local_Score_i_2', 'Local_Score_i_avg',
        'Local_Area_A', 'Local_Area_B', 'Local_Area_i_1', 'Local_Area_i_2', 'Local_Area_i_avg', 
        'Total_pae_A', 'Total_pae_B', 'Total_pae_i_1', 'Total_pae_i_2', 'Total_pae_i_avg',
        'Local_pae_A', 'Local_pae_B', 'Local_pae_i_1', 'Local_pae_i_2', 'Local_pae_i_avg',
        'Rank', 'saved folder', 'pdb'
    ]

    data = {
        'ID' : ID,
        'pLDDT': round(plddt, 2),
        'pTM': ptm,
        'ipTM': iptm,
        'Total_pae_A': round(pae_protein_a, 2),
        'Total_pae_B': round(pae_protein_b, 2),
        'Total_pae_i_1': round(pae_interaction1, 2),
        'Total_pae_i_2': round(pae_interaction2, 2),
        'Total_pae_i_avg': round(pae_interaction_total, 2),
        'Local_pae_A': round(average_selected_protein_a, 2),
        'Local_pae_B': round(average_selected_protein_b, 2),
        'Local_pae_i_1': round(average_selected_interaction1, 2),
        'Local_pae_i_2': round(average_selected_interaction2, 2),
        'Local_pae_i_avg': round(average_selected_interaction_total, 2),
        'Local_Score_A': round(average_selected_protein_a_score, 3),
        'Local_Score_B': round(average_selected_protein_b_score, 3),
        'Local_Score_i_1': round(average_selected_interaction1_score, 3),
        'Local_Score_i_2': round(average_selected_interaction2_score, 3),
        'Local_Score_i_avg': round(average_selected_interaction_total_score, 3),
        'Local_Area_A': local_interaction_protein_a,
        'Local_Area_B': local_interaction_protein_b,
        'Local_Area_i_1': local_interaction_interface_1,
        'Local_Area_i_2': local_interaction_interface_2,
        'Local_Area_i_avg': local_interaction_interface_avg,
        'Rank': rank,
        'saved folder': os.path.dirname(pdb_file_path),  # Gets the parent directory of the file path
        'pdb': os.path.basename(pdb_file_path),  # Extracts just the base name of the pdb file,
        'protein_a_len': protein_a_len,
        'protein_b_len': protein_b_len,
    }

    df = pd.DataFrame(data, index=[file_name])[COLUMNS_ORDER]
    df["Relaxed"] = df['pdb'].str.contains("_relaxed")
    return df


def reverse_and_scale_matrix(matrix: np.ndarray, pae_cutoff: float = 12.0) -> np.ndarray:
    """
    Scale the values in the matrix such that:
    0 becomes 1, pae_cutoff becomes 0, and values greater than pae_cutoff are also 0.
    
    Args:
    - matrix (np.ndarray): Input numpy matrix.
    - pae_cutoff (float): Threshold above which values become 0.
    
    Returns:
    - np.ndarray: Transformed matrix.
    """
    
    # Scale the values to [0, 1] for values between 0 and cutoff
    scaled_matrix = (pae_cutoff - matrix) / pae_cutoff
    scaled_matrix = np.clip(scaled_matrix, 0, 1)  # Ensures values are between 0 and 1
    
    return scaled_matrix

def process_pdb_files(directory_path: str, processed_files=[], pae_cutoff: float = 12.0, name_separator: str = "___") -> pd.DataFrame:
    '''
    Processes PDB files within a directory (directory_path)
    '''
    all_files = os.listdir(directory_path)
    pdb_files = [f for f in all_files if f.endswith(".pdb") and f not in processed_files]
    print('PDB files in this directory: ', len(pdb_files))

    # Start with an empty DataFrame with columns explicitly defined if necessary
    df_results = pd.DataFrame()  
    results_list = []  # Use a list to collect data frames or series

    # Iterate through pdb files
    for pdb_file in pdb_files:
        pdb_file_path = os.path.join(directory_path, pdb_file)
        try:
            # call calculate_pae to get the PAE result
            result = calculate_pae(pdb_file_path, print_results=False, pae_cutoff=pae_cutoff, name_separator=name_separator)
            if result is not None:
                results_list.append(result)
        except FileNotFoundError:
            print(f"Error: File {pdb_file_path} not found. Skipping...")

    if results_list:
        df_results = pd.concat(results_list, axis=0, ignore_index=True)
    return df_results

def analyze(base_path, saving_base_path, cutoff, folders_to_analyze, num_processes, name_separator: str = "___"):
    '''
    Processes multiple data folders to apply a cutoff operation and saves the results.

    Args:
        base_path (str): The path to the directory containing data folders.
        saving_base_path (str): The path where the processed data should be saved.
        cutoff (float): The threshold value used for processing data.
        folders_to_analyze (list): A list of folder names within the base directory to be analyzed.
        num_processes (int): The number of parallel processes to utilize for processing data.
        name_separator (str, optional): A delimiter used in naming the output files. Defaults to '___'.

    Returns:
        None: This function does not return any value but saves the processed data in the specified path.
    '''
    # Make save directory if it doesn't exist 
    if not os.path.exists(saving_base_path):
        os.makedirs(saving_base_path)

    # loop through data files
    # all of these are going to be concatenated into ONE output file. 
    output_filename = f"pae_{cutoff}_alphafold_analysis.csv"
    full_saving_path = os.path.join(saving_base_path, output_filename)
    print(f"Saving to {full_saving_path}")
    
    # delete what's there if needed
    if clear_old_outputs:
        print("Clearing past outputs...")
        path_to_clear = f"{saving_base_path}/pae_{cutoff}_alphafold_analysis"
        if os.path.exists(f"{path_to_clear}.csv"): os.remove(f"{path_to_clear}.csv")
        if os.path.exists(f"{path_to_clear}_unrelaxed_processed.csv"): os.remove(f"{path_to_clear}_unrelaxed_processed.csv")
        if os.path.exists(f"{path_to_clear}_relaxed_processed.csv"): os.remove(f"{path_to_clear}_relaxed_processed.csv")
        
    full_data = None
    # Iterate through each subfolder
    for i, data_folder in enumerate(folders_to_analyze):
        # Create the path where we'll pull data from this folder
        directory_path = os.path.join(base_path, data_folder)
        print(f"\nProcessing data from {directory_path}")
        
        # If the path exists (it should eist)
        if os.path.exists(directory_path):
            # Check for existing processed files
            if os.path.exists(full_saving_path):
                try:
                    existing_df = pd.read_csv(full_saving_path)
                    processed_files = existing_df['pdb'].tolist() if 'pdb' in existing_df.columns else []
                except EmptyDataError:
                    # Handle the empty file situation
                    print(f"File {full_saving_path} is empty. Starting from scratch.")
                    existing_df = pd.DataFrame()
                    processed_files = []
            # No processed file exists yet, start a new one
            else:
                existing_df = pd.DataFrame()
                processed_files = []

            new_data = process_pdb_files(directory_path, processed_files, cutoff, name_separator)

            # Combine old and new data only if there's new data
            if not new_data.empty:
                combined_df = pd.concat([existing_df, new_data])
                
                # Add the combined DataFrame to the full_data DataFrame
                if i==0:
                    full_data = combined_df
                else:
                    full_data = pd.concat([full_data,combined_df])
            else:
                print(f"No new data to append. CSV remains unchanged.")

        else:
            print(f"Directory {directory_path} does not exist! Skipping...")
    # if this file isn't None - we have something to write
    if full_data is not None:
        # sort by binder number - just makes it easier to look at!
        # full_data['number'] = full_data['ID'].str.split('t',expand=True)[1].str.split('_',expand=True)[0].astype(int)
        # pdb.set_trace()
        full_data['number'] = full_data['ID']
        full_data = full_data.sort_values(by=['number','Rank','Relaxed'],ascending=True).reset_index(drop=True).drop(columns='number')
        full_data.to_csv(full_saving_path, index=False)
        print(f"Saved processed data to {full_saving_path}")


def process_pdb_file(pdb_file, directory_path, processed_files, cutoff, name_separator):
    pdb_file_path = os.path.join(directory_path, pdb_file)
    print("\nProcessing:", pdb_file)
    try:
        results = calculate_pae(pdb_file_path, False, cutoff, name_separator)
        return results
    except FileNotFoundError:
        print(f"Error: File {pdb_file_path} not found. Skipping...")
        return None

def process_pdb_files_parallel(directory_path: str, processed_files=[], pae_cutoff: float = 12.0, num_processes=1) -> pd.DataFrame:
    all_files = os.listdir(directory_path)
    pdb_files = [f for f in all_files if f.endswith(".pdb") and f not in processed_files]

    df_results = pd.DataFrame()
    
    with Pool(num_processes) as pool:
        results = pool.starmap(calculate_pae, [(os.path.join(directory_path, f), False, pae_cutoff) for f in pdb_files])
        for res in results:
            if res is not None:
                df_results = df_results.append(res, ignore_index=True)
    
    return df_results

def get_subdirectories(base_path):
    """
    Get a list of subdirectories in the given base path.

    Parameters:
    - base_path: The path where to look for subdirectories.

    Returns:
    - List of subdirectories as strings.
    """
    return [d.name for d in os.scandir(base_path) if d.is_dir()]

def get_num_cpu_cores():
    try:
        return os.cpu_count() or 1
    except AttributeError:
        return multiprocessing.cpu_count() or 1

def extract_top_results(df):
    """
    Process the given DataFrame to extract rank1 rows and compute average values.
    Returns a new DataFrame with both rank1 and average information.
    """

    # Drop rows with any NaN values in the relevant columns
    df = df.dropna(subset=['Local_Score_i_avg', 'Local_Area_i_avg', 'ipTM', 'pTM', 'pLDDT'])
    
    # Extract rank 1 rows and rename columns accordingly
    rank1_rows = df[df['Rank'] == 1][['ID', 'Local_Score_i_avg', 'Local_Area_i_avg', 'ipTM', 'pTM', 'pLDDT', 'pdb','Relaxed']].copy()
    rank1_rows.rename(columns={
        'Local_Score_i_avg': 'Best LIS',
        'Local_Area_i_avg': 'Best LIA',
        'ipTM': 'Best ipTM',
        'pTM': 'Best pTM',
        'pLDDT': 'Best pLDDT'
    }, inplace=True)

    # Group by ID
    average_values = df.groupby('ID', as_index=False)[['Local_Score_i_avg', 'Local_Area_i_avg', 'ipTM', 'pTM', 'pLDDT']].mean()
    average_values.rename(columns={
        'Local_Score_i_avg': 'Average LIS',
        'Local_Area_i_avg': 'Average LIA',
        'ipTM': 'Average ipTM',
        'pTM': 'Average pTM',
        'pLDDT': 'Average pLDDT'
    }, inplace=True)
    
    # Merge rank 1 rows with the average values using 'pae_file_name' as the key
    final_df = pd.merge(rank1_rows, average_values, on='ID', how='left')
    # Best LIS ≥ 0.203 AND Best LIA ≥ 3432, or Average LIS ≥ 0.073 AND Average LIA ≥ 1610.
    final_df['Positive PPI'] = (
    ((final_df['Best LIS'] >= 0.203) & (final_df['Best LIA'] >= 3432)) |
    ((final_df['Average LIS'] >= 0.073) & (final_df['Average LIA'] >= 1610))
    )

    # Define the columns of interest and their order
    columns_of_interest = [
        'ID', 'Positive PPI',
        'Best ipTM','Best pTM',
        'Best pLDDT',
        'Best LIS', 'Average LIS',
        'Best LIA', 'Average LIA',
        'Average ipTM',
        'Average pTM',
        'Average pLDDT',
        'pdb',
        'Relaxed'
    ]

    final_df = final_df[columns_of_interest]

    return final_df

def main(base_path, pae_cutoff=12):
    '''
    Processes the outputs in base_path. If base
    '''
    start_time = time.time()
    
    num_processes = get_num_cpu_cores()
    print(f"Number of available CPU cores: {num_processes}")

    # Define your base paths, cutoff value, and folder list
    name_separator = "___"  # add separator that distinguishes protein_1 and protein_2
    
    saving_base_path = f"/home/tc415/muPPIt/muppit/fold_test_results"

    # Generate folders_to_analyze list
    folders_to_analyze = [""]
    # if 'output_' not in base_path: # if we didn't specify an ouptut folder - we're working with a top folder - then we should find the subs
    #     folders_to_analyze = get_subdirectories(base_path)
    #     print('Iterating through these subfolders: ', ','.join(folders_to_analyze))

    # Call the analyze (previously: main) function with the folder list
    analyze(base_path, saving_base_path, pae_cutoff, folders_to_analyze, num_processes, name_separator)

    # Path to the specific folder where the original files are located
    folder_path = saving_base_path  

    # Path to the folder where you want to save the processed files
    saving_path = saving_base_path

    # Ensure the saving path directory exists, if not, create it
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    file_names = [f for f in os.listdir(folder_path) if f.endswith("alphafold_analysis.csv")]

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        print(file_name)

        try:
            # pdb.set_trace()
            df = pd.read_csv(file_path)

            # Check if DataFrame is empty
            if df.empty:
                print(f"File {file_name} is empty. Skipping...")
                continue

            # Process the DataFrame
            processed_df_unrelaxed= extract_top_results(df.loc[df["Relaxed"]==False])
            processed_df_unrelaxed['number'] = processed_df_unrelaxed['ID']
            processed_df_unrelaxed = processed_df_unrelaxed.sort_values(by='number',ascending=True).reset_index(drop=True).drop(columns='number')
            # processed_df_relaxed = extract_top_results(df.loc[df["Relaxed"]==True])
            # processed_df_relaxed['number'] = processed_df_relaxed['ID']
            # processed_df_relaxed = processed_df_relaxed.sort_values(by='number',ascending=True).reset_index(drop=True).drop(columns='number')

            # Constructing the new file name
            base_name = base_path.split('/')[-1].strip()
            new_file_path_unrelaxed = os.path.join(saving_path, f"{base_name}_unrelaxed_processed.csv")
            # new_file_path_relaxed = os.path.join(saving_path, f"{base_name}_relaxed_processed.csv")

            # Save the processed DataFrame to a new file
            # pdb.set_trace()
            processed_df_unrelaxed.to_csv(new_file_path_unrelaxed, index=False)
            print(f"Processed {file_name} and saved to {new_file_path_unrelaxed}")
            # processed_df_relaxed.to_csv(new_file_path_relaxed, index=False)
            # print(f"Processed {file_name} and saved to {new_file_path_relaxed}")

        except EmptyDataError:
            print(f"File {file_path} is empty and was skipped.")
        except Exception as e:  # General exception to catch unexpected errors
            print(f"Error processing {file_name}: {str(e)}")

    # print("Processing completed. Took {:.2f}s".format(time.time() - start_time))
    # # open up the file with the relaxed final stats 
    subprocess.run(["code", "-r", f"{new_file_path_unrelaxed}"])

# CALL MAIN TO DO FULL PROCESSING
main(base_path)
#print(get_subdirectories(base_path))