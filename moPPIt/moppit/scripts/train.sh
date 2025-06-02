export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
torchrun --nproc_per_node=6 train_bindevaluator.py -o train_bindevaluator_0 \
-lr 1e-3 \
-n_layers 8 \
-d_model 128 \
-d_hidden 128 \
-n_head 8 \
-d_inner 64 \
-batch_size 32 \
--max_epochs 30 \
--dropout 0.3 \
--grad_clip 0.5 \
--kl_weight 0.1

#-sm '/home/tc415/muPPIt/muppit/model_path/continue_train_base_1/model-epoch=19-val_loss=0.33.ckpt' \

#export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
#torchrun --nproc_per_node=6 --master_port=25431 finetune.py -o finetune_7 \
#-sm '/home/tc415/muPPIt/muppit/model_path/optimize_train_4/model-epoch=29-val_mcc=0.59.ckpt' \
#-lr 1e-3 \
#-n_layers 8 \
#-d_model 128 \
#-n_head 8 \
#-d_inner 64 \
#-batch_size 32 \
#--max_epochs 50 \
#-dropout 0.3 \
#--grad_clip 0.5 \
#--kl_weight 0.5
