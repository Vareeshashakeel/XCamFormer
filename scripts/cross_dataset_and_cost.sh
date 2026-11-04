#!/bin/bash
# Cross-dataset evaluation (Table 10) and embedding cost (Table 12) with the MARS-trained model.
# Usage: bash scripts/cross_dataset_and_cost.sh path/to/Mars_xcamformer.pth
source "$(dirname "$0")/common.sh"
M=${1:?MARS checkpoint}; mkdir -p feats results
python3 analysis/extract_features.py --Dataset_name iLIDSVID --model_path "$M" --out feats/mars2ilids.pt
python3 analysis/extract_features.py --Dataset_name PRID     --model_path "$M" --out feats/mars2prid.pt
python3 analysis/extract_features.py --Dataset_name Mars     --model_path "$M" --out feats/mars_full.pt
python3 analysis/embedding_cost.py --feats feats/mars_full.pt --out results/embedding_mars
