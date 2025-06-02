## PhosphositePTM
Phosphorylation site prediction

### Usage
#### Configs
The `config.py` script holds configurations for **training** and **plotting**. 

**Training configs:**
```python
# Models to benchmark (bool)
BENCHMARK_ESM3B = False                                     # True if you want to benchmark ESM3B model
BENCHMARK_ESM650M = False                                   # True if you want to benchmark ESM650M model
BENCHMARK_MAMBA = False                                     # True if you want to benchmark Mamba model
BENCHMARK_ONEHOT = True                                     # True if you want to benchmark OneHot model
# Number of replicates
N_REPLICATES = 2                                            # How many times to train each model (results are averaged for plotting)
# GPU settings
CUDA_VISIBLE_DEVICES="0,1"                                  # str to queue up GPUs. leave blank "" if you don't want to set this variable
```

**Plotting configs**
```python
ESM3B_METRICS_PATH = ''                                     # /path/to/ESM3B/results. Leave blank "" if you don't want to plot this
ESM650M_METRICS_PATH = ''                                   # /path/to/ESM650M/results Leave blank "" if you don't want to plot this
MAMBA_METRICS_PATH = ''                                     # /path/to/PTM Mamba/results. Leave blank "" if you don't want to plot this
ONEHOT_METRICS_PATH = ''                                    # /path/to/One Hot/results. Leave blank "" if you don't want to plot this

PLOT_SAVE_DIR = ''                                          # /path/to/where/you/want/plots/saved
```

#### Training
The `train.py` script generates embeddings for the input sequences using the specified models from `config.py`. Then, it trains a classifier on the embeddings. Finally, it calls methods from `plot.py` to automatically generate plots for the models you just trained. All of your results are stored in "/PhosphositePTM/results/timestamp", where **timestamp** is a unique string encoding today's date and time. This folder also contains `train_settings.txt`, which stores your config settings for this run. 

To run, enter in terminal:
```bash
python train.py
```
or, to run the (sometimes long) training process in the background:
```bash
nohup python train.py > train.out 2> train.err &
```

#### Plotting
The `plot.py` script generates the bar chart and line charts from the PTM-Mamba paper for any models of your choosing (specified in `config.py`). The plots themselves, as well as the dataframes used to create them, will be saved in the path you have specified. 

To run, enter in terminal:
```bash
python plot.py
```