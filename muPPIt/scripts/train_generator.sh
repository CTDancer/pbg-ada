export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_API_KEY=b7b6ecceb6854bd12f58809f18264f979509d13b

torchrun --nproc_per_node=4 --master_port=54321 train_generator.py \
-lr 1e-4 \
-batch_size 4 \
-num_heads 4 \
-num_layers 4 \
-max_epochs 10 \
-dropout 0.1 \
-grad_clip 32 \
-k 5 \
-passes 20 \
-o '/home/tc415/muPPIt/checkpoints/generator_1'