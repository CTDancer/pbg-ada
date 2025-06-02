import random
from pepmlm import generate_peptide
from predict import calculate_score, PeptideModel
from argparse import ArgumentParser
import numpy as np
import pdb
import torch

AA = 'ARNDCEQGHILKMFPSTWYV'


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

    return result


def cal_score(binder_seq, protein_seq, motif, model, args):
    prediction, threshold = calculate_score(protein_seq, binder_seq, model, args)
    binding_score = 0
    binding_site_score = []
    for pos in motif:
        binding_score += prediction[pos]
        binding_site_score.append({pos: prediction[pos].item()})
    print(f"Binding Site Score = {binding_site_score}")

    penalty = []
    penalty_sites = []
    for i in range(len(prediction)):
        if i not in motif and prediction[i] >= threshold:
            penalty.append(prediction[i])
            penalty_sites.append({i: prediction[i].item()})

    print(f"Penalty Sites = {penalty_sites}")

    penalty_score = sum(penalty) / len(motif)  # sum(penalty) / len(penalty) * len(penalty) / len(motif)
    score = len(motif) - binding_score + penalty_score
    print(f"Binder:{binder_seq}, Score:{score}")

    # pdb.set_trace()

    return score


def main(args):
    random.seed(args.seed)
    found = False
    binders = []
    generation = 0

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    model = PeptideModel.load_from_checkpoint(args.sm,
                                              n_layers=args.n_layers,
                                              d_model=args.d_model,
                                              n_head=args.n_head,
                                              d_k=64,
                                              d_v=128,
                                              d_inner=64).to(device)

    binder_seq = args.binder_seq
    protein_seq = 'CHHRICHCSNRVFLCQESKVTEIPSDLPRNAIELRFVLTKLRVIQKGAFSGFGDLEKIEISQNDVLEVIEADVFSNLPKLHEIRIEKANNLLYINPEAFQNLPNLQYLLISNTGIKHLPDVHKIHSLQKVLLDIQDNINIHTIERNSFVGLSFESVILWLNKNGIQEIHNSAFNGTQLDELNLSDNNNLEELPNDVFHGASGPVILDISRTRIHSLPSYGLENLKKLRARSTYNLKKLPTLEKLVALMEASLTYPSHCCAFANWRRQISELHPICNKSILRQEVDYMTQARGQRSSLAEDNESSYSRGFDMTYTEFDXDLCNEVVDVTCSPKPDAFNPCEDIMGYNILR'
    motif = [15, 16, 32, 34, 36, 37, 58, 60, 61, 81, 83, 85, 86, 108, 110, 111, 128, 134, 135, 154, 156, 158, 160, 161, 178, 179, 184, 203, 204, 224, 225, 227, 247, 313, 316, 319]
    cal_score(binder_seq, protein_seq, motif, model, args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--binder_seq', type=str, required=True)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--num_binders', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("-sm", default='/home/tc415/muPPIt/muppit/model_path/finetune_base_10/model-epoch=29-val_mcc=0.55.ckpt',
                        help="File containing initial params", type=str)
    parser.add_argument("-batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-n_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("-d_model", type=int, default=64, help="Dimension of model")
    parser.add_argument("-n_head", type=int, default=6, help="Number of heads")
    parser.add_argument("-d_inner", type=int, default=64)
    args = parser.parse_args()
    main(args)



