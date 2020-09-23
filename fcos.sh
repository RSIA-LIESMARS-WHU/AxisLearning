python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=$((RANDOM + 10000)) \
        tools/train_net.py \
        --skip-test \
        --config-file configs/fcos/fcos_R_50_FPN_1x.yaml \
        DATALOADER.NUM_WORKERS 2 \
        OUTPUT_DIR training_dir/fcos_R_50_FPN_1x
