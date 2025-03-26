python main.py \
    --use_color \
    --use_normal \
    --detector detector_Vote2Cap_DETRv2 \
    --checkpoint_dir pretrained/Vote2Cap_DETRv2_XYZ_COLOR_NORMAL \
    --gpu 3 \
    --max_epoch 1080 \
    --batchsize_per_gpu 8