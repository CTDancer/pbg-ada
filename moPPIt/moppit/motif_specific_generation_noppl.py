import random
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pepmlm import generate_peptide, compute_pseudo_perplexity
from predict_bindevaluator import calculate_score, PeptideModel
from argparse import ArgumentParser
import numpy as np
import pdb
import torch
import os

AA = 'ARNDCEQGHILKMFPSTWYV'

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


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
    score = 0
    # for pos in motif:
    #     if prediction[pos] < 0.5:
    #         score += 5
    #
    # penalty = []
    # for i in range(len(prediction)):
    #     if i not in motif and prediction[i] >= 0.5:
    #         score += 0.5

    for pos in motif:
        if prediction[pos] >= 0.5:
            score -= 1
    for i in range(len(prediction)):
        if i not in motif and prediction[i] >= 0.5:
            score += 1

    return score


class Binder(object):
    def __init__(self, binder_seq, model, pepmlm, tokenizer, args):
        self.binder_seq = binder_seq
        self.protein_seq = args.protein_seq
        self.motif = parse_motif(args.motif.strip('[]'))
        self.model = model
        self.args = args
        self.pepmlm = pepmlm
        self.tokenizer = tokenizer
        self.score = cal_score(binder_seq, self.protein_seq, self.motif, self.model, self.args)

    def mutated_aa(self):
        global AA
        mutated_aa = random.choice(AA)
        return mutated_aa

    def mutate_seq(self, binder_seq):
        position = random.randint(0, len(binder_seq) - 1)
        mutated_seq = binder_seq[:position] + self.mutated_aa() + binder_seq[position + 1:]
        return mutated_seq

    def mate(self, par2):
        assert len(par2.binder_seq) == len(self.binder_seq)

        child = ''
        for i in range(len(par2.binder_seq)):
            a1 = self.binder_seq[i]
            a2 = par2.binder_seq[i]

            prob = random.random()
            if prob < 0.45:
                child += a1
            elif prob < 0.90:
                child += a2
            else:
                child += self.mutated_aa()

        return Binder(child, self.model, self.pepmlm, self.tokenizer, args)

    def simulated_annealing(self, T_max, T_min, alpha):
        def accept(delta_score, T):
            delta_score = delta_score.item()
            if delta_score < 0:
                return True
            else:
                r = random.random()
                if r < np.exp(-delta_score / T):
                    return True
                else:
                    return False

        T = T_max
        score = self.score
        binder_seq = self.binder_seq

        print(f"Initial Binder Sequence: {binder_seq}")

        while T > T_min and score > 1:
            mutated_seq = self.mutate_seq(binder_seq)

            new_score = cal_score(mutated_seq, self.protein_seq, self.motif, self.model, self.args)

            delta_score = new_score - score

            if accept(delta_score, T):
                binder_seq = mutated_seq
                score = new_score
                print(f"New Sequence: {binder_seq}, Score: {score}")

            T = T * alpha
            print(f"New T: {T}")

        return binder_seq, score


def main(args):
    print(parse_motif(args.motif.strip('[]')))
    random.seed(args.seed)
    binders = []
    generation = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = PeptideModel.load_from_checkpoint(args.sm,
                                              n_layers=args.n_layers,
                                              d_model=args.d_model,
                                              d_hidden=args.d_hidden,
                                              n_head=args.n_head,
                                              d_k=64,
                                              d_v=128,
                                              d_inner=64).to(device)

    tokenizer = AutoTokenizer.from_pretrained("ChatterjeeLab/PepMLM-650M")
    pepmlm = AutoModelForMaskedLM.from_pretrained("ChatterjeeLab/PepMLM-650M").to(device)

    temp_binders = []
    count = 5
    while len(temp_binders) < args.num_binders and count >= 0:
        # pdb.set_trace()
        temp_binders += generate_peptide(args.protein_seq, args.peptide_length, args.top_k, args.num_binders-len(temp_binders))['Binder'].tolist()
        temp_binders = [seq for seq in temp_binders if 'X' not in seq]
        temp_binders = list(set(temp_binders))
        count -= 1

    print(f"Pool Size = {len(temp_binders)}")

    # pdb.set_trace()
    for binder_seq in temp_binders:
        binders.append(Binder(binder_seq, model, pepmlm, tokenizer, args))

    binders = sorted(binders, key=lambda binder: binder.score)
    print(
        f"Generation: -1\tBinder: {binders[0].binder_seq}\tScore: {binders[0].score}")

    no_improvement_generations = 0
    max_tolerance = 20
    previous_score = 10000
    previous_ppl = 10000
    # threshold = int(0.1*len(parse_motif(args.motif.strip('[]'))))
    threshold = -1000
    # if threshold == 0:
    #     threshold = 0.5     # a little relaxation
    print(f"Threshold: {threshold}")

    while no_improvement_generations < max_tolerance:
        previous_score = binders[0].score
        if binders[0].score <= threshold:
            break

        new_binders = []

        s = int((10*args.num_binders)/100)
        new_binders.extend(binders[:s])

        s = int((90*args.num_binders)/100)
        half = int(args.num_binders / 2)
        for _ in range(s):
            par1 = random.choice(binders[:half])
            par2 = random.choice(binders[:half])
            new_binders.append(par1.mate(par2))

        new_binders = sorted(new_binders, key=lambda binder: binder.score)

        if new_binders[0].score < previous_score:
            no_improvement_generations = 0
            print(
                f"Generation: {generation}\tBinder: {new_binders[0].binder_seq}\tScore: {new_binders[0].score}")
        else:
            no_improvement_generations += 1
            print(
                f"Generation: {generation}\tNo improvement {no_improvement_generations} generations")

        binders = new_binders
        generation += 1

    print(f"Generation: {generation}\tBinder: {binders[0].binder_seq}\tScore: {binders[0].score}")
    #
    # print("Start Simulated Annealing!")
    # final_binder, final_score = binders[0].simulated_annealing(10, 0.001, 0.95)
    # print(f"Final Binder: {final_binder}, Final Score: {final_score}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--protein_seq', type=str, required=True)
    parser.add_argument('--peptide_length', type=int, required=True)
    parser.add_argument('--motif', type=str, required=True)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--num_binders', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("-sm", default='/home/tc415/muPPIt/muppit/train_base_1/model-epoch=14-val_loss=0.40.ckpt',
                        help="File containing initial params", type=str)
    parser.add_argument("-batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-n_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("-d_model", type=int, default=64, help="Dimension of model")
    parser.add_argument("-d_hidden", type=int, default=128, help="Dimension of CNN block")
    parser.add_argument("-n_head", type=int, default=6, help="Number of heads")
    parser.add_argument("-d_inner", type=int, default=64)
    args = parser.parse_args()
    main(args)
