import torch
from types import SimpleNamespace
import numpy as np
from utils import TrainProcessor
import random
from torch_geometric.loader import DataLoader
from model import GNNTrans
import os
import argparse

random.seed(42)
np.random.seed(42)

esm_embedding_dims = {
    'esm2_t48_15B_UR50D': 5120,
    'esm2_t36_3B_UR50D': 2560,
    'esm2_t33_650M_UR50D': 1280,
    'esm2_t30_150M_UR50D': 640,
    'esm2_t12_35M_UR50D': 480,
    'esm2_t6_8M_UR50D': 320
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model using embeddings.')
    parser.add_argument('--embedding_type', type=str, required=True, choices=['ptm_mamba', 'esm'],
                        help='Type of embeddings to use: "ptm_mamba" or "esm".')
    parser.add_argument('--esm_model', type=str, choices=esm_embedding_dims.keys(),
                        help='ESM model to use if embedding_type is "esm".')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory containing processed data.')
    parser.add_argument('--result_dir', type=str, default=None, help='Directory to save results.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on.')

    args = parser.parse_args()

    if args.embedding_type == 'ptm_mamba':
        data_dir = args.data_dir if args.data_dir else "processed3_ptm_mamba"
        result_dir = args.result_dir if args.result_dir else "ptm_mamba_results"
        input_dim = 768
    elif args.embedding_type == 'esm':
        if not args.esm_model:
            raise ValueError("You must specify an ESM model when using ESM embeddings.")
        data_dir = args.data_dir if args.data_dir else "processed3_esm"
        result_dir = args.result_dir if args.result_dir else f"esm_{args.esm_model}_results"
        input_dim = esm_embedding_dims[args.esm_model]
    
    os.makedirs(result_dir, exist_ok=True)

    train_args = {
        'epochs': 500,
        'batch_size': 64,
        'device': args.device,
        'opt': 'adam',
        'opt_scheduler': 'step',
        'opt_decay_step': 20,
        'opt_decay_rate': 0.92,
        'weight_decay': 1e-4,
        'lr': 3e-5,
        'es_patience': 20,
        'save': True
    }
    train_args = SimpleNamespace(**train_args)
    print(train_args)

    len_ls = [11, 15, 21, 25, 31, 35, 41, 45, 51, 55, 61]
    for seq_len in len_ls:
        train_ls, val_ls, test_ls = torch.load(f'{data_dir}/{seq_len}.pt')
        train_data_loader = DataLoader(train_ls, batch_size=train_args.batch_size)
        val_data_loader = DataLoader(val_ls, batch_size=train_args.batch_size)
        test_data_loader = DataLoader(test_ls, batch_size=train_args.batch_size)

        for i in range(10):
            model = GNNTrans(input_dim=input_dim, hidden_dim=256, num_layers=2)
            model.to(train_args.device)
            print(model)

            train_val = TrainProcessor(
                model=model,
                loaders=[train_data_loader, val_data_loader, test_data_loader],
                args=train_args
            )
            best_model, test_metrics = train_val.train()
            print('test loss: {:5f}; test acc: {:4f}; test auroc: {:4f}; test auprc: {:.4f}'.format(
                test_metrics.loss, test_metrics.acc, test_metrics.auroc, test_metrics.auprc))

            if train_args.save:
                save_dir = f'./{result_dir}/{seq_len}'
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                save_path = os.path.join(save_dir,
                                        '{}_acc{:.4f}_roc{:.4f}_prc{:.4f}_f1{:.4}_mcc{:.4f}_precision{:.4f}_recall{:.4f}.pt'.format(
                                            i, test_metrics.acc, test_metrics.auroc, test_metrics.auprc, test_metrics.f1, test_metrics.mcc, test_metrics.precision, test_metrics.recall))
                torch.save(best_model.state_dict(), save_path)
