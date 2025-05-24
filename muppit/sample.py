import os
import hydra
import lightning as L
import numpy as np
import omegaconf
import pandas as pd
import rdkit
import rich.syntax
import rich.tree
import torch
from tqdm.auto import tqdm
import pdb

import dataloader
import diffusion
from models.bindevaluator import BindEvaluator

rdkit.rdBase.DisableLog('rdApp.error')

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)
omegaconf.OmegaConf.register_new_resolver(
  'if_then_else',
  lambda condition, x, y: x if condition else y
)

def _print_config(
    config: omegaconf.DictConfig,
    resolve: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.

  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style,
                        guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)

def parse_motif(motif: str) -> list:
    parts = motif.split(',')
    result = []

    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))

    result = [pos-1 for pos in result]
    print(f'Target Motifs: {result}')
    return torch.tensor(result)

@hydra.main(version_base=None, config_path='./configs',
            config_name='config')
def main(config: omegaconf.DictConfig) -> None:
  # Reproducibility
  L.seed_everything(config.seed)
  os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
  torch.use_deterministic_algorithms(True)
  torch.backends.cudnn.benchmark = False

#   _print_config(config, resolve=True)
  print(f"Checkpoint: {config.eval.checkpoint_path}")

  tokenizer = dataloader.get_tokenizer(config)
  target_sequence = tokenizer(config.eval.target_sequence, return_tensors='pt')['input_ids']
  
  pretrained = diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config, logger=False)
  pretrained.eval()

  bindevaluator = BindEvaluator.load_from_checkpoint(
    config.guidance.classifier_checkpoint_path,
    n_layers=8,
    d_model=128,
    d_hidden=128,
    n_head=8,
    d_k=64,
    d_v=128,
    d_inner=64)

  samples = []
  for _ in tqdm(
      range(config.sampling.num_sample_batches),
      desc='Gen. batches', leave=False):
    sample = pretrained.sample(
      target_sequence = target_sequence,
      target_motifs = parse_motif(config.eval.target_motifs),
      classifier_model = bindevaluator
    )
    # print(f"Batch took {time.time() - start:.2f} seconds.")
    # 6 - 12
    # 6 - 49
    samples.extend(
      pretrained.tokenizer.batch_decode(sample))

    print([sample.replace(' ', '')[5:-5] for sample in samples])
  
  samples = [sample.replace(' ', '')[5:-5] for sample in samples]
  print(samples)

if __name__ == '__main__':
  main()