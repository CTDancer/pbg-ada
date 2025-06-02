#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
torchrun --nproc_per_node=6 --master_port=25142 finetune.py -o finetune_correct_1 \
-sm '/home/tc415/moPPIt/moppit/model_path/optimize_train_4/model-epoch=29-val_mcc=0.59.ckpt' \
-lr 1e-3 \
-n_layers 8 \
-d_model 128 \
-n_head 8 \
-d_inner 64 \
-batch_size 32 \
--max_epochs 50 \
-dropout 0.5 \
--grad_clip 0.5 \
--kl_weight 1
#
# /home/tc415/muPPIt/muppit/model_path/optimize_train_4/model-epoch=29-val_mcc=0.59.ckpt
# /home/tc415/muPPIt/muppit/finetune_3/model-epoch=20-val_mcc=0.51-val_loss=0.28.ckpt

#export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
#
#pid_of_command1=3310282
#command2='torchrun --nproc_per_node=6 train_evaluator.py -o optimize_train_2 -lr 1e-4 -n_layers 6 -d_model 64 -n_head 6 -d_inner 64 -batch_size 64 --max_epochs 30 --dropout 0.2 --grad_clip 0.5 --kl_weight 0.5'
#
#while kill -0 $pid_of_command1 2> /dev/null; do
#    sleep 5
#done
#
#eval "$command2"