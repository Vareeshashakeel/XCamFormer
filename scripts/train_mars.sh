#!/bin/bash
# XCamFormer on MARS (Table 3: 120 epochs, batch 64, lambda_x 0.15, lambda_p 0.10, gamma_x 0.25, gamma_p 0.10, tau 0.07, blocks 5,8)
source "$(dirname "$0")/common.sh"
python3 VID_Trans_ReID.py --Dataset_name Mars --model_path "$VIT" --output_dir ./outputs/mars \
  --epochs 120 --eval_every 10 --batch_size 64 --num_workers 4 --seq_len 4 \
  --center_w 0.0005 --attn_w 1.0 \
  --xcam_w 0.15 --xcam_temp 0.07 --xcam_same_cam_w 0.25 --xcam_blocks 5,8 \
  --part_xcam_w 0.10 --part_xcam_temp 0.07 --part_xcam_same_cam_w 0.10
