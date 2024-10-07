export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc-per-node 2 train.py \
    --model_path initial-vi-gemma-2b \
    --local_data_path vi-wiki
