export CUDA_VISIBLE_DEVICES=7
export WANDB_API_KEY=b7b6ecceb6854bd12f58809f18264f979509d13b

python -u predict_decoder_2.py \
-lr 1e-4 \
-batch_size 8 \
-d_node 1296 \
-n_heads 8 \
-n_layers 6 \
-d_ff 2048 \
-max_epochs 30 \
-grad_clip 0.5 \
-sm '/home/tc415/muPPIt/checkpoints/train_decoder_0/model-epoch=12-val_loss=0.00.ckpt' \
-s 'VAPITKVKKKQQLGKFRIHLE'