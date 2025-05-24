import pandas as pd
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

for i in range(6, 50):
    df = pd.read_csv(f'/home/tc415/discrete-diffusion-guidance/samples/step_128/{i}.csv')
    seqs = df['sequence'].tolist()

    outliers = []
    for seq in seqs:
        tok = tokenizer(seq)['input_ids']
        if len(tok) - 2 != i:
            outliers.append(seq)
    
    print(f"{i}.csv: {outliers}")
