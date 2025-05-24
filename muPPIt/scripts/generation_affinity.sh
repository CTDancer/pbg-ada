export CUDA_VISIBLE_DEVICES=7

python -u generator.py \
-d_node_muppit 256 \
-d_edge_muppit 128 \
-d_cross_edge_muppit 128 \
-d_position_muppit 8 \
-n_heads_muppit 2 \
-n_intra_layers_muppit 2 \
-n_mim_layers_muppit 2 \
-n_cross_layers_muppit 2 \
-sm_muppit '/home/tc415/muPPIt/checkpoints/good/muppit.ckpt' \
-d_node_decoder 1296 \
-n_layers_decoder 6 \
-n_heads_decoder 8 \
-d_ff_decoder 2048 \
-dropout_decoder 0.2 \
-sm_decoder '/home/tc415/muPPIt/checkpoints/train_decoder_0/model-epoch=12-val_loss=0.00.ckpt' \
-num_updates 300 \
-step_size 0.5 \
-top_k 3 \
-num_binders 50 \
-binder_length 20 \
-wt 'MASETFEFQAEITQLMSLIINTVYSNKEIFLRELISNASDALDKIRYKSLSDPKQLETCWWLPIWITPCPCIKVLEIRDSGIGMTKAELINNLGTIAKSGTKAFMEALSAGADVSMIGQFGVGFYSLFLVADRVQVISKSNDDEQYIWESNAGGSFTVTLWCHWCWIGRGTILRLFLKDDQLEYLECKRIIEVGWRHSEFVADWGIGGNFFCRC' \
-mut 'MASETFEFQAEITQLMSLIINTVYSNKEIFLRELISNASDALDKIRYKSLSDPKQLETEPDLFIRITPKPEQKVLEIRDSGIGMTKAELINNLGTIAKSGTKAFMEALSAGADVSMIGQFGVGFYSLFLVADRVQVISKSNDDEQYIWESNAGGSFTVTLDEVNERIGRGTILRLFLKDDQLEYLEEKRIKEVIKRHSEFVAYPIQLVVTKEVE'
