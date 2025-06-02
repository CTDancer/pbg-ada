export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

torchrun --nproc_per_node=6 test_bindevaluator.py \
-sm '/home/tc415/moPPIt/moppit/model_path/finetune_bindevaluator_0/model-epoch=30-val_mcc=0.60-val_loss=0.51.ckpt' \
-n_layers 8 \
-d_model 128 \
-d_hidden 128 \
-n_head 8 \
-d_inner 64 \
-batch_size 32 \
--kl_weight 1

#/home/tc415/muPPIt/muppit/model_path/optimize_train_4/model-epoch=29-val_mcc=0.59.ckpt
#/home/tc415/muPPIt/muppit/finetune_correct_0/model-epoch=39-val_mcc=0.58-val_loss=0.52.ckpt
# /home/tc415/muPPIt/muppit/model_path/finetune_bindevaluator_0/model-epoch=30-val_mcc=0.60-val_loss=0.51.ckpt
# /home/tc415/muPPIt/muppit/model_path/train_bindevaluator_0/model-epoch=29-val_mcc=0.61.ckpt