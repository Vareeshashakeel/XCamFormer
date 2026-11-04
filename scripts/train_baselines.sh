#!/bin/bash
# Camera-removed VID-Trans-ReID baseline (E2 / "w/o XCam"): same settings with both cross-camera losses switched off.
# Usage: bash scripts/train_baselines.sh Mars|iLIDSVID|PRID
source "$(dirname "$0")/common.sh"
D=${1:?dataset: Mars, iLIDSVID or PRID}
if [ "$D" = "Mars" ]; then S="--epochs 120 --eval_every 10 --batch_size 64 --num_workers 4"; else S="--epochs 160 --eval_every 20 --batch_size 16 --num_workers 2 --split_id 0"; fi
python3 VID_Trans_ReID.py --Dataset_name $D --model_path "$VIT" --output_dir ./outputs/baseline_$D $S \
  --seq_len 4 --center_w 0.0005 --attn_w 1.0 --xcam_w 0 --part_xcam_w 0
