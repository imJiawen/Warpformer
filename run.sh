# task select from ['mor', 'los', 'decom', 'wbm', 'vent', 'vaso', 'active', 'physio']
# --dp_flag: use DataParallel

# small scale MIMIC-III

# python Main_warp.py \
#     --data_path /path/to/datasets/ \
#     --d_model 32 --batch 32 --n_head 1 --n_layers 2 --d_k 8 --d_v 8 --lr 1e-3 --epoch 2 --patience 5 \
#     --log /path/to/log/ --save_path /path/to/save/ \
#     --task 'mor' --seed 0 --dp_flag --warp_num '0_1' 


# large scale  MIMIC-III
# for large sacle, --load_in_batch is required

# python Main_warp.py \
#     --data_path /path/to/datasets/ \
#     --d_model 32 --batch 16 --n_head 1 --n_layers 2 --d_k 8 --d_v 8 --lr 1e-3 --epoch 40 --patience 5 \
#     --log /path/to/log/ \
#     --save_path /path/to/save/ \
#     --task 'wbm' --seed 0 --dp_flag --warp_num '0_1' --load_in_batch


# PhysioNet (median len. 72)

python Main_warp.py \
    --data_path /path/to/datasets/ \
    --batch 32 --lr 1e-3 --epoch 50 --patience 5 \
    --log /path/to/log/ --save_path /path/to/save/ \
    --task 'physio' --seed 0 --warp_num '0_0.2_1' \
    --batch_size 32 --d_inner_hid 32 --d_k 8 --d_model 32 --d_v 8 \
    --dropout 0.0 --n_head 1 --n_layers 2 


# Human Activity (median len. 50)
# for human activity, perform per time point classification

# python Main_warp.py \
#     --data_path /path/to/datasets/ \
#     --batch 64 --lr 1e-3 --epoch 50 --patience 10 \
#     --log /path/to/log/ \
#     --save_path /path/to/save/ \
#     --task 'active' --seed 0 --warp_num '0_1.2_1' 