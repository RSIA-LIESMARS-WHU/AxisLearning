python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net.py \
    --skip-test \
    --config-file configs/rfcos/rfcos_R_101_FPN_2x_fcos_range.yaml \
    DATALOADER.NUM_WORKERS 2 \
    OUTPUT_DIR training_dir/$1
