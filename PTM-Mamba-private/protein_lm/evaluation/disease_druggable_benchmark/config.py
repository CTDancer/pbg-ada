#### Training settings for disease/druggable benchmark
# GPU settings
CUDA_VISIBLE_DEVICES="1"                                    # str to queue up GPUs. leave blank if you don't want to set this variable

# Models to benchmark (bool)
BENCHMARK_ESM={
    "esm2_t48_15B_UR50D": False,                            # True to benchmark ESM2-15B
    "esm2_t36_3B_UR50D": True,                             # True to benchmark ESM2-3B
    "esm2_t33_650M_UR50D": True,                            # True to benchmark ESM2-650M
    "esm2_t30_150M_UR50D": False,                           # True to benchmark ESM2-150M
    "esm2_t12_35M_UR50D": False,                            # True to benchmark ESM2-35M
    "esm2_t6_8M_UR50D": False                               # True to benchmark ESM2-8M
}
BENCHMARK_ONEHOT=True
BENCHMARK_ONEHOTPTM=True
BENCHMARK_MAMBA=True
BENCHMARK_PTM_TRANSFORMER =True
BENCHMARK_MAMBA_SAPROT = True

# Number of replicates
N_REPLICATES = 5                                            # How many times to train each model

# Benchmark tests to perform
DRUGGABILITY=True                                           # True to perform Druggability benchmark
DISEASE=True                                                # True to perform Disease benchmark

#### Summary and plotting settings for disease/druggable benchmark
# Druggability file paths
DRUGGABILITY_RESULTS_DIR = 'results/temp'
DISEASE_RESULTS_DIR = 'results/temp'