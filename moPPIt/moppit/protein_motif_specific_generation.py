import random
from predict import calculate_score, PeptideModel
from progen.progen2.sample import generate_proteins
from progen.progen2.likelihood import compute_loglikelihood
from argparse import ArgumentParser
import numpy as np
import pdb
import torch
from progen.progen2.models.progen.modeling_progen import ProGenForCausalLM
from tokenizers import Tokenizer

AA = 'ARNDCEQGHILKMFPSTWYV'


def create_model(ckpt, fp16=True):
    if fp16:
        return ProGenForCausalLM.from_pretrained(ckpt, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
        return ProGenForCausalLM.from_pretrained(ckpt)


def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())

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
    for pos in motif:
        binding_score += prediction[pos]

    penalty = []
    for i in range(len(prediction)):
        if i not in motif and prediction[i] >= threshold:
            penalty.append(prediction[i])

    penalty_score = sum(penalty) / len(motif)  # sum(penalty) / len(penalty) * len(penalty) / len(motif)
    score = len(motif) - binding_score + penalty_score
    # print(f"Binder:{binder_seq}, Score:{score}")

    return score


def cal_loglikelihood(binder_seq, model, tokenizer, args):
    args.context = f"1{binder_seq}2"
    loglikelihood = compute_loglikelihood(args, model, tokenizer)
    return loglikelihood


class Binder(object):
    def __init__(self, binder_seq, model, progen, progen_tokenizer, args):
        self.binder_seq = binder_seq
        self.protein_seq = args.protein_seq
        self.motif = parse_motif(args.motif.strip('[]'))
        self.model = model
        self.progen = progen
        self.progen_tokenizer = progen_tokenizer
        self.args = args
        self.score = cal_score(self.binder_seq, self.protein_seq, self.motif, self.model, self.args)
        self.loglikelihood = cal_loglikelihood(self.binder_seq, self.progen, self.progen_tokenizer, self.args)

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

        return Binder(child, self.model, self.progen, self.progen_tokenizer, self.args)

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
    random.seed(args.seed)
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

    temp_binders = generate_proteins(args)

    device = torch.device(args.device)
    ckpt = f'/home/tc415/muPPIt/muppit/progen/progen2/checkpoints/{args.model}'

    progen = create_model(ckpt=ckpt, fp16=args.fp16).to(device)

    progen_tokenizer = create_tokenizer_custom(file='/home/tc415/muPPIt/muppit/progen/progen2/tokenizer.json')

    for binder_seq in temp_binders:
        binders.append(Binder(binder_seq, model, progen, progen_tokenizer, args))

    no_improvement_generations = 0
    max_tolerance = 50
    previous_score = 10000

    while no_improvement_generations < max_tolerance:
        binders = sorted(binders, key=lambda binder: (-binder.loglikelihood, binder.score))

        if binders[0].score <= 1:
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

        if new_binders[0].score < previous_score:
            no_improvement_generations = 0
            print(f"Generation: {generation}\tBinder: {new_binders[0].binder_seq}\tFitness: {new_binders[0].score}\tLogLikelihood: {new_binders[0].loglikelihood}")
        else:
            no_improvement_generations += 1
            print(f"Generation: {generation}\tNo improvement {no_improvement_generations} generations\tLogLikelihood: {new_binders[0].loglikelihood}")

        previous_score = binders[0].score
        binders = new_binders
        generation += 1

    print(f"Generation: {generation}\tBinder: {binders[0].binder_seq}\tFitness: {binders[0].score}\tLogLikelihood: {binders[0].loglikelihood}")

    # print("Start Simulated Annealing!")
    # final_binder, final_score = binders[0].simulated_annealing(10, 0.001, 0.99)
    # print(f"Final Binder: {final_binder}, Final Score: {final_score}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--protein_seq', type=str, required=True)
    parser.add_argument('--binder_length', type=int, required=True)
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
    parser.add_argument("-n_head", type=int, default=6, help="Number of heads")
    parser.add_argument("-d_inner", type=int, default=64)

    models_151M = ['progen2-small']
    models_754M = ['progen2-medium', 'progen2-oas', 'progen2-base']
    models_2B = ['progen2-large', 'progen2-BFD90']
    models_6B = ['progen2-xlarge']
    models = models_151M + models_754M + models_2B + models_6B
    parser.add_argument('--model', type=str, choices=models, default='progen2-large')
    parser.add_argument('--device', type=str, default='cuda:7')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--p', type=float, default=0.95)
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--fp16', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--context', type=str, default='1')
    parser.add_argument('--sanity', default=True, type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()
    args.max_length = args.binder_length + 1
    main(args)
