#### Training settings for disease/druggable benchmark
# GPU settings
CUDA_VISIBLE_DEVICES="0"                                    # str to queue up GPUs. leave blank if you don't want to set this variable

# Models to benchmark (bool)
BENCHMARK_ESM={
    "esm2_t48_15B_UR50D": False,                            # True to benchmark ESM2-15B
    "esm2_t36_3B_UR50D": False,                             # True to benchmark ESM2-3B
    "esm2_t33_650M_UR50D": True,                            # True to benchmark ESM2-650M
    "esm2_t30_150M_UR50D": False,                           # True to benchmark ESM2-150M
    "esm2_t12_35M_UR50D": False,                            # True to benchmark ESM2-35M
    "esm2_t6_8M_UR50D": False                               # True to benchmark ESM2-8M
}
BENCHMARK_MAMBA=True                                        # True to benchmark PTM-Mamba
BENCHMARK_PTM_TRANSFORMER = True                            # True to benchmark PTM Transformer architecture
BENCHMARK_MAMBA_SAPROT = True                               # True to benchmark Mamba with SaProt embedings


# Number of replicates
N_REPLICATES = 10                                           # How many times to train each model

##### SUMMARY SETTING
RESULTS_DIR = 'results/test_results'                        # directory where the results are