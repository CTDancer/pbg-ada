export CUDA_VISIBLE_DEVICES=2

python -u generation_BDLoss.py \
-d_node_muppit 256 \
-d_edge_muppit 128 \
-d_cross_edge_muppit 128 \
-d_position_muppit 8 \
-n_heads_muppit 2 \
-n_intra_layers_muppit 2 \
-n_mim_layers_muppit 2 \
-n_cross_layers_muppit 2 \
-sm_muppit '/home/tc415/muPPIt/checkpoints/train_affinity_8/model-epoch=23-val_acc=0.65.ckpt' \
-d_node_decoder 1296 \
-output_dim_decoder 24 \
-n_layers_decoder 6 \
-n_heads_decoder 8 \
-d_ff_decoder 2048 \
-sm_decoder '/home/tc415/muPPIt/checkpoints/decoder/model-epoch=08-val_loss=0.00.ckpt' \
-num_updates 100 \
-step_size 500 \
-top_k 3 \
-num_binders 50 \
-binder_length 12 \
-mut 'ESQPDPMPDDLHKSSEFTGTMGNMKYLYDDHYVSATKVKSVDKFLAHDLIYNINDKKLNNYDKVKTELLNEDLANKYKDEVVDVYGSNYYVNCYFSSKDNVGKVTSGKTCMYGGITKHEGNHFDNGNLQNVLIRVYENKRNTISFEVQTDKKSVTAQELDIKARNFLINKKNLYEFNSSPYETGYIKFIESNGNTFWYDMMPAPGDKFDQSKYLMIYKDNKMVDSKSVKIEVHLTTKNG' \
-wt 'ESQPDPMPDDLHKSSEFTGTMGNMKYLYDDHYVSATKVKSVDKFLAHDLIYNINDKKLNNYDKVKTELLNEDLANKYKDEVVDVYGSNYAVNCYFSSKDNVGKVTSGKTCMYGGITKHEGNHFDNGNLQNVLIRVYENKRNTISFEVQTDKKSVTAQELDIKARNFLINKKNLYEFNSSPYETGYIKFIESNGNTFWYDMMPAPGDKFDQSKYLMIYKDNKMVDSKSVKIEVHLTTKNG'