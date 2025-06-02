import pandas as pd
import io
import random
import requests
from Bio import SeqIO
import math
import os
import pickle
import sys
import pdb
import numpy as np
import torch
import esm
from argparse import ArgumentParser
from torch.nn import LogSoftmax
from torch import softmax

from model import SALTnPEPPR
from data import ProteinPeptideDataModule

def main(seq):
  model_file = '/home/tc415/muPPIt/muppit/saltnpeppr/production_version_epoch=3.ckpt'

  model = SALTnPEPPR.load_from_checkpoint(model_file, map_location=torch.device('cpu'))
  model.to('cpu')

  snp_alphabet = esm.Alphabet.from_architecture('ESM-1b')
  snp_batch_converter = snp_alphabet.get_batch_converter()

  def get_AA_SnP_scores(protein_seq):
    protein_tuple = [(1, protein_seq)]
    batch_labels, batch_strs, batch_tokens = snp_batch_converter(protein_tuple)

    with torch.no_grad():
      scores = model(batch_tokens, None)

    softmax = torch.nn.Softmax(dim=1) # turns raw scores to probabilities 
    scores = softmax(scores.detach()) # calculate probabilities of not binding (0th index) or binding (1st index)

    npscores = scores[:,1].detach().numpy() # take only column 1 - probability of binding for this position
    return npscores

  npscores = get_AA_SnP_scores(seq)
  print(npscores)
  pdb.set_trace()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-s', type=str, required=True)

    args = parser.parse_args()
    main(args.s)