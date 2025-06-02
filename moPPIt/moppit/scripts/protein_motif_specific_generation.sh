export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python protein_motif_specific_generation.py \
-sm '/home/tc415/muPPIt/muppit/model_path/continue_train_base_1/model-epoch=19-val_loss=0.33.ckpt' \
--protein_seq 'PRLCYLVKEGGSYGFSLKTVQGKKGVYMTDITPQGVAMRAGVLADDHLIEVNGENVEDASHEEVVEKVKKSGSRVMFLLVDKKITKF' \
--binder_length 98 \
--motif '[7,10-20,28,29,31,60,61,63-65,67,68]' \
--top_k 3 \
--num_binders 100 \
--t 0.8 \
--p 0.9
