#!/bin/bash
# XCamFormer on iLIDSVID (Table 3: 160 epochs, batch 16, lambda_x 0.10, lambda_p 0.08, gamma_x 0.25, gamma_p 0.10, tau 0.07, blocks 5,8)
source "$(dirname "$0")/common.sh"
python3 VID_Trans_ReID.py --Dataset_name iLIDSVID --model_path "$VIT" --output_dir ./outputs/ilidsvid --split_id 0 \
  --epochs 160 --eval_every 20 --batch_size 16 --num_workers 2 --seq_len 4 \
  --center_w 0.0005 --attn_w 1.0 \
  --xcam_w 0.10 --xcam_temp 0.07 --xcam_same_cam_w 0.25 --xcam_blocks 5,8 \
  --part_xcam_w 0.08 --part_xcam_temp 0.07 --part_xcam_same_cam_w 0.10
