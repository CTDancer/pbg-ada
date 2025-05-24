export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_API_KEY=b7b6ecceb6854bd12f58809f18264f979509d13b

torchrun --nproc_per_node=4 --master_port=54321 train_delta_3.py \
-lr 1e-4 \
-batch_size 2 \
-d_node 256 \
-d_edge 128 \
-d_cross_edge 128 \
-d_position 8 \
-n_heads 2 \
-n_intra_layers 2 \
-n_mim_layers 2 \
-n_cross_layers 2 \
-max_epochs 30 \
-grad_clip 2 \
-sm '/home/tc415/muPPIt/checkpoints/pretrained/muppit.ckpt' \
-o '/home/tc415/muPPIt/checkpoints/train_delta_2'