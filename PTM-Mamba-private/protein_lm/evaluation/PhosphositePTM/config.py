#### Training configs: when you train, plots will automatically be generated for everything you just trained.
# Models to benchmark (bool)
BENCHMARK_ESM3B = False                                     # True if you want to benchmark ESM3B model
BENCHMARK_ESM650M = False                                   # True if you want to benchmark ESM650M model
BENCHMARK_MAMBA = False                                     # True if you want to benchmark Mamba model
BENCHMARK_PTM_TRANSFORMER = True                            # True if you want to benchmark PTM Transformer architecture
BENCHMARK_MAMBA_SAPROT = True                               # True if you want to benchmark Mamba with SaProt embedings
BENCHMARK_ONEHOT = True                                     # True if you want to benchmark OneHot model

# Number of replicates
N_REPLICATES = 2                                            # How many times to train each model

# GPU settings
CUDA_VISIBLE_DEVICES="1"                                    # str to queue up GPUs. leave blank if you don't want to set this variable

#### Plotting configs: if you just want to make new plots, specify paths here.
# Leave a path blank '' or None if you do not want to plot it 

ESM3B_METRICS_PATH = 'five_trials/jul22_2024/esm3b_test_metrics.csv'
ESM650M_METRICS_PATH = 'five_trials/jul22_2024/esm650m_test_metrics.csv'
MAMBA_METRICS_PATH = 'five_trials/jul22_2024/mamba_test_metrics.csv'
ONEHOT_METRICS_PATH = 'five_trials/jul22_2024/one_hot_test_metrics.csv'
PTM_TRANSFORMER_METRICS_PATH = ''
MAMBA_SAPROT_METRICS_PATH = ''

PLOT_SAVE_DIR = '' # /path/to/where/you/want/plot/saved

