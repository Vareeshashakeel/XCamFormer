#!/bin/bash
# Run-to-run variation on PRID2011 (Table 8): baseline and XCamFormer with several seeds, plus single-objective runs.
source "$(dirname "$0")/common.sh"
C="--Dataset_name PRID --model_path $VIT --output_dir ./outputs/seeds --split_id 0 --epochs 160 --eval_every 20 --batch_size 16 --num_workers 2"
for s in 1234 1 2 3; do
  python3 VID_Trans_ReID.py $C --seed $s --run_name full_seed$s --xcam_w 0.10 --part_xcam_w 0.08
  python3 VID_Trans_ReID.py $C --seed $s --run_name base_seed$s --xcam_w 0 --part_xcam_w 0
done
python3 VID_Trans_ReID.py $C --run_name iccs_only  --xcam_w 0.10 --part_xcam_w 0
python3 VID_Trans_ReID.py $C --run_name plccs_only --xcam_w 0    --part_xcam_w 0.08
python3 analysis/summarize_runs.py --group "full=outputs/seeds/full_seed*_results.json" --group "base=outputs/seeds/base_seed*_results.json" --out outputs/seeds/summary.md
