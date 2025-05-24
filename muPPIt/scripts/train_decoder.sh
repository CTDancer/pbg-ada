export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_API_KEY=b7b6ecceb6854bd12f58809f18264f979509d13b

torchrun --nproc_per_node=4 --master_port=54321 train_decoder_2.py \
-lr 1e-4 \
-batch_size 8 \
-d_node 1296 \
-n_heads 8 \
-n_layers 6 \
-d_ff 2048 \
-max_epochs 30 \
-grad_clip 0.5 \
-o '/home/tc415/muPPIt/checkpoints/train_decoder_0'
