"""Screening evaluation"""
import sys
# sys.path.append('/home/yz927/.local/lib/python3.8/site-packages')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import pickle
import glob
import os
import subprocess
import multiprocessing

def run_esm_fold(fasta_file, pdb_output_path, device, num_recycles, max_tokens_per_batch, chunk_size, cpu_only, cpu_offload):
    # Set CUDA_VISIBLE_DEVICES environment variable
    # Prepare ESM-Fold command with device argument
    cmd = [
        "/home/yz927/.local/bin/esm-fold",
        "-i", fasta_file,
        "-o", pdb_output_path,
        # "--device", str(device),  # Pass the device index as an integer
        "--num-recycles", str(num_recycles),
        "--max-tokens-per-batch", str(max_tokens_per_batch)
    ]
    if chunk_size is not None:
        cmd.extend(["--chunk-size", str(chunk_size)])
    if cpu_only:
        cmd.append("--cpu-only")
    if cpu_offload:
        cmd.append("--cpu-offload")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device+4)

    # Run ESM-Fold
    subprocess.run(cmd, env=env)

def process_files(files, device, pdb_output_path, num_recycles, max_tokens_per_batch, chunk_size, cpu_only, cpu_offload):
    for fasta_file in files:
        run_esm_fold(fasta_file, pdb_output_path, device, num_recycles, max_tokens_per_batch, chunk_size, cpu_only, cpu_offload)

def main():
    # os.environ['PATH'] += os.pathsep + "/home/yz927/.local/bin"
    # print("Current Environment:", os.environ['PATH'])
    # try:
    #     subprocess.run(["esm-fold", "--help"], check=True)
    # except FileNotFoundError as e:
    #     print(f"Error: {e}")
    fasta_path = "/home/tc415/muPPIt/dataset/fasta_files"
    pdb_output_path = "/home/tc415/muPPIt/dataset/ppiref_pdbs"
    # pickle_files_directory = "/container_mount_storage/yz927/projects/protllama2_output/param_search"
    num_recycles = 4
    max_tokens_per_batch = 2048
    chunk_size = 128
    cpu_only = False
    cpu_offload = False

    all_fasta_files = [os.path.join(fasta_path, f) for f in os.listdir(fasta_path) if f.endswith('.fasta')]

    total_files = len(all_fasta_files)
    part_size = total_files // 4
    fasta_files_parts = [all_fasta_files[i*part_size:(i+1)*part_size] for i in range(4)]
    # Ensure all files are included, especially if total_files is not divisible by 4
    for i in range(total_files % 4):
        fasta_files_parts[-1].append(all_fasta_files[-(i+1)])
    # Run ESM-Fold for the sequences in the low homology set

    with multiprocessing.Pool(processes=4, maxtasksperchild=None) as pool:  # Adjust the number of processes to match the number of GPUs
        pool_results = []
        for i in range(4):
            pool_results.append(pool.apply_async(process_files, (fasta_files_parts[i], i, pdb_output_path, num_recycles, max_tokens_per_batch, chunk_size, cpu_only, cpu_offload)))
        
        for result in pool_results:
            result.get()

if __name__ == "__main__":
    main()
