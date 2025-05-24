export CUDA_VISIBLE_DEVICES=5,6,7
export WANDB_API_KEY=b7b6ecceb6854bd12f58809f18264f979509d13b

torchrun --nproc_per_node=3 --master_port=54321 train_correct.py \
-lr 1e-4 \
-batch_size 2 \
-d_node_muppit 256 \
-d_edge_muppit 128 \
-d_cross_edge_muppit 128 \
-d_position_muppit 8 \
-n_heads_muppit 2 \
-n_intra_layers_muppit 2 \
-n_mim_layers_muppit 2 \
-n_cross_layers_muppit 2 \
-node_dim_decoder 256 \
-edge_dim_decoder 128 \
-hidden_dim_decoder 256 \
-n_layers_decoder 4 \
-dropout_decoder 0.2 \
-max_epochs 30 \
-grad_clip 2 \
-margin 2 \
-sm_muppit '/home/tc415/muPPIt/checkpoints/pretrained/muppit.ckpt' \
-o '/home/tc415/muPPIt/checkpoints/train_correct_0'