export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 test_evaluator.py \
-sm '/home/tc415/muPPIt/muppit/model_path/optimize_train_4/model-epoch=29-val_mcc=0.59.ckpt' \
-n_layers 8 \
-d_model 128 \
-n_head 8 \
-d_inner 64 \
-batch_size 32 \
--kl_weight 0.1

#/home/tc415/muPPIt/muppit/model_path/optimize_train_4/model-epoch=29-val_mcc=0.59.ckpt
#/home/tc415/muPPIt/muppit/finetune_correct_0/model-epoch=39-val_mcc=0.58-val_loss=0.52.ckpt