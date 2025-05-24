export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=b7b6ecceb6854bd12f58809f18264f979509d13b

python -u test_affinity.py \
-lr 1e-4 \
-batch_size 1 \
-d_node 256 \
-d_edge 128 \
-d_cross_edge 128 \
-d_position 8 \
-n_heads 2 \
-n_intra_layers 2 \
-n_mim_layers 2 \
-n_cross_layers 2 \
-max_epochs 5 \
-grad_clip 0.5 \
-sm '/home/tc415/muPPIt/checkpoints/train_affinity_9/model-epoch=24-val_acc=0.64.ckpt' 