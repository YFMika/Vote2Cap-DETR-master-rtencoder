python scst_tuning.py \
    --use_color \
    --use_normal \
    --dataset scene_scanrefer \
    --base_lr 1e-6 \
    --detector detector_Vote2Cap_DETR \
    --captioner captioner_dcc \
    --freeze_detector \
    --use_beam_search \
    --batchsize_per_gpu 2 \
    --max_epoch 180 \
    --pretrained_captioner exp_scanrefer/Vote2Cap_DETR_RGB_NORMAL/checkpoint_best.pth \
    --checkpoint_dir exp_scanrefer/scst_Vote2Cap_DETR_RGB_NORMAL