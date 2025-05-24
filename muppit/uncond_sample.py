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
import csv

import dataloader
import diffusion

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

def parse_range(tgt_range: str) -> list:
    parts = tgt_range.split(',')
    result = []

    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))

    return result

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
  
  pretrained = diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config, logger=False)
  pretrained.eval()

  target_lengths = parse_range(config.model.length_range)
  
  for length in target_lengths:
    config.model.length = length + 2
    samples = []
    for _ in tqdm(
        range(config.sampling.num_sample_batches),
        desc='Gen. batches', leave=False):
        sample = pretrained.sample()
        # print(f"Batch took {time.time() - start:.2f} seconds.")
        samples.extend(
        pretrained.tokenizer.batch_decode(sample))

        # print([sample.replace(' ', '')[5:-5] for sample in samples])
    
    samples = [sample.replace(' ', '')[5:-5] for sample in samples]
    print(samples)

    df = pd.DataFrame(samples, columns=['sequence'])
    df.to_csv(f'/home/tc415/discrete-diffusion-guidance/samples/step_{config.sampling.steps}/{length}.csv', index=False)

if __name__ == '__main__':
  main()
