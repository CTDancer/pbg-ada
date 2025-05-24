export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 train.py \
-lr 1e-5 \
-batch_size 2 \
-d_node 256 \
-d_edge 256 \
-d_cross_edge 256 \
-d_position 8 \
-n_heads 4 \
-n_intra_layers 2 \
-n_mim_layers 2 \
-n_cross_layers 2 \
-max_epochs 30 \
-grad_clip 0.5 \
-delta 1 \
-o '/home/tc415/muPPIt/checkpoints/debug'