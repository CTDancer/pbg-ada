## TransPTM
A Transformer-Based Model for Non-Histone Acetylation Site Prediction


### Dataset
Download the `dataset.csv` at [here](https://drive.google.com/file/d/1BmxdbQkTzobPDujy3m-X27MEsc-hP9Zc/view?usp=drive_link) and put it in the CWD.


### Usage
#### Configs
The `config.py` script holds configurations for **training** and **summarization (including plotting)**. 

**Training configs:***
```python
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
```
**Summarization and plotting configs:**
You can generate a few types of output files after training classification benchmarking files: summary tables, bar charts, and line charts. Each replicate of the model is evaluated on a held-out test set during training, resulting in the calculation of Loss, Accuracy, Precision, Recall, F1, MCC, AUPRC, and AUROC. Additionally, the model is evaluated on batches with differing sequence length ranges. Different types of tables and plots are made by averaging test metrics (1) across replicates for each sequence length, (2) across sequence length for each replicate, and (3): across replicates for each sequence length, and then across sequence length, yielding a final average.


```python
RESULTS_DIR = 'results/jan1_10234'                          # /path/to/results/directory/from/training
```

#### Training
The `train.py` script generates embeddings for the input sequences using the specified PTM-Mamba, ESM, and other models from `config.py`. Then, it trains a classifier on the embeddings. Finally, it calls methods from `summary.py` script to automatically generate plots and summary .csv files for the models. All of your results are stored in "/TransPTM/results/timestamp", where **timestamp** is a unique string encoding today's date and time. This folder also contains `train_settings.txt`, which stores your config settings for this run. 

The embeddings are stored within TransPTM, and are structured as follows:
```
TransPTM/
└── embeddings/
└── processed_data/
```

Where **embeddings** and **processed_data** contain subfolders for each model. When you benchmark a certain model (e.g. PTM-Mamba) for the first time, its embeddings and processed data will be stored in **embeddings/ptm_mamba** and **processed_data/ptm_mamba** and never calculated again. In future benchmarks, embeddings and processed data will be loaded from these cached locations. This should save you time when repeating benchmarks.


The output is within TransPTM, and is structured as follows:

```
results/
└── timestamp/
    ├── figures/
    │   ├── averages_only/
    │   │   ├── Accuracy.png
    │   │   ├── AUPRC.png
    │   │   ├── AUROC.png
    │   │   ├── avg_metrics.png
    │   │   ├── F1.png
    │   │   ├── Loss.png
    │   │   ├── MCC.png
    │   │   ├── metrics_grid.png
    │   │   ├── Precision.png
    │   │   └── Recall.png
    │   └── individual_points_plotted/
    │       ├── Accuracy_with_replicates.png
    │       ├── AUPRC_with_replicates.png
    │       ├── AUROC_with_replicates.png
    │       ├── avg_metrics_with_replicate_avgs.png
    │       ├── avg_metrics_with_seqlen_avgs.png
    │       ├── F1_with_replicates.png
    │       ├── Loss_with_replicates.png
    │       ├── MCC_with_replicates.png
    │       ├── metrics_grid_across_replicates.png
    │       ├── Precision_with_replicates.png
    │       └── Recall_with_replicates.png
    ├── test_metrics/
    ├── summarized_test_results_across_reps.csv
    ├── summarized_test_results_across_seqlens.csv
    ├── summarized_test_results.csv
    └── test_results.csv
```
- **timestamp**: a unique string encoding today's date and time (e.g. jan_1_1023).
- **figures**: a folder holding all generated figures. Each figure shows all models that were benchmarked.
- **averages_only**: a folder holding plots that only show average values at each xtick. These plots may be more readable. `avg_metrics.png` is a bar chart with each test metric averaged across replicate AND sequence length. Any file with a metric name, like `AUROC.png`, is a line plot containing the average performance for that metric, across replicates, on each sequence length. `metrics_grid.png` summarizes all the individual metric plots onto one plot for easier viewing.
- **individual_points_plotted**: a folder holding plots that show average values at each xtick, as well as *individual points* representing the values that were averaged. These plots may be less readable, but more informative. `avg_metrics_with_replicate_avgs.png` is a bar chart where bars contain the same meaning as in `avg_metrics.png`. The dots represent averages across sequence length for each replicate (n_dots = n_replicates). `avg_metrics_with_seqlen_avgs.png` is a bar chart where bars contain the same meaning as in `avg_metrics.png`. The dots represent averages across replicates for each sequence length (n_dots = n_seq_lens = 11). Any file with a metric name, like `AUROC_with_replicates.png`, is a line plot containing the average performance for that metric, across replicates, on each sequence length. Dots on these plots represent the values at that sequene length for each replicate (n_dots = n_replicates). `metrics_grid_across_replicates.png` summarizes all the individual metric plots onto one plot for easier viewing.
- **summarized_test_results_across_reps.csv**: contains average test metrics per sequence length per model. Averages were taken across replicates for each sequence length.
- **summarized_test_results_across_seqlens.csv**: contains average test metrics per replicate per model. Averages were taken across sequence lengths for each replicate.
- **summarized_test_results.csv**: contains average test metrics per model. Averages were taken across replicates and sequence lengths. 

To run, enter in terminal:
```bash
python train.py
```
or, to run the (sometimes long) training process in the background:
```bash
nohup python train.py > train.out 2> train.err &
```

#### Plotting and Summary
The `summary.py` script generates the bar chart and line charts from the PTM-Mamba paper for any results directory of your choosing (specified in `config.py`). If you'd like to mix and match results from different training runs, you simply need to combine the rows you want from their separate `results/test_results.csv` files, and save your new `test_results.csv` file into a directory. Then, specify this directory in `config.py`. 

The plots themselves, as well as the dataframes used to create them, will be saved in the path you have specified. 

To run, enter in terminal:
```bash
python summary.py
```