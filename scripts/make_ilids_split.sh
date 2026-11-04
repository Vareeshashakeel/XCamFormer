#!/bin/bash
# Recreate the iLIDS-VID train/test split used for the reported results from the released checkpoint.
# Usage: bash scripts/make_ilids_split.sh path/to/iLIDSVID_partxcam_best.pth
: "${ILIDS_ROOT:?set ILIDS_ROOT to the iLIDS-VID folder}"
export ILIDS_ROOT
cd "$(dirname "$0")/.."
python3 analysis/recover_old_split.py --model_path "${1:?iLIDS-VID checkpoint}"
