python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=$((RANDOM + 20000)) \
    tools/train_net.py \
    --skip-test \
    --config-file configs/rfcos/best_hrsc.yaml \
    DATALOADER.NUM_WORKERS 2 \
    OUTPUT_DIR training_dir/$1
