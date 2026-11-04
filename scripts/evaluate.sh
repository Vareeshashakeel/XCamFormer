#!/bin/bash
# Evaluate a trained checkpoint (Tables 4-6). Add --flip_test for the flip-test row of Table 4.
# Usage: bash scripts/evaluate.sh Mars|iLIDSVID|PRID path/to/checkpoint.pth [--flip_test]
source "$(dirname "$0")/common.sh"
python3 VID_Test_Final.py --Dataset_name ${1:?dataset} --model_path ${2:?checkpoint} ${3}
