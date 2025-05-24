export CUDA_VISIBLE_DEVICES=2
export WANDB_API_KEY=b7b6ecceb6854bd12f58809f18264f979509d13b

python predict_generator.py \
-lr 1e-4 \
-batch_size 4 \
-num_heads 4 \
-num_layers 4 \
-dropout 0.1 \
-k 5 \
-passes 15 \
-sm '/home/tc415/muPPIt/checkpoints/generator_1/epoch=00-val_loss=2.92-val_acc=0.09.ckpt' \
-target 'DTPDAGASFSRHFAANFLDVFGEEVRRVLV' \
-max_length 20