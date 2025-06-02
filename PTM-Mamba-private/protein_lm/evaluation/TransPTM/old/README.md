## TranPTM
A Transformer-Based Model for Non-Histone Acetylation Site Prediction


### Dataset
Download the `dataset.csv` at [here](https://drive.google.com/file/d/1BmxdbQkTzobPDujy3m-X27MEsc-hP9Zc/view?usp=drive_link) and put it in the CWD.


### Usage
the `train.py` script generates embeddings for the input sequences using the specified PTM-Mamba or ESM model and trains a classifier on the embeddings. The `summary.py` script summarizes the results of the trained models.

For PTM-Mamba (specify the `--ptm_mamba_ckpt /path/to/ptm_mamba.ckpt`):
```bash
python train.py --input_csv dataset.csv --embed_dir ptm_mamba_emb --data_dir processed3_ptm_mamba --result_dir ptm_mamba_results --device cuda:0 --ptm_mamba_ckpt /path/to/ptm_mamba.ckpt --embedding_type ptm_mamba
```
For ESM-2-650M:
```bash
python train.py --input_csv dataset.csv --embed_dir esm_emb --data_dir processed3_esm --result_dir esm_esm2_t33_650M_UR50D_results --device cuda:0 --esm_model esm2_t33_650M_UR50D --embedding_type esm
```
You can replace the --esm_model argument with any of the other ESM model names as needed:

- esm2_t48_15B_UR50D
- esm2_t36_3B_UR50D
- esm2_t30_150M_UR50D
- esm2_t12_35M_UR50D
- esm2_t6_8M_UR50D

Summarize the results:

```bash
python summary.py
```
