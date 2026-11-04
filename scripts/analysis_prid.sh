#!/bin/bash
# Camera-invariance, layer-wise and t-SNE analyses (Table 9, Figs. 7-8) for a baseline / XCamFormer pair on PRID2011.
# Usage: bash scripts/analysis_prid.sh path/to/baseline.pth path/to/xcamformer.pth
source "$(dirname "$0")/common.sh"
B=${1:?baseline checkpoint}; X=${2:?XCamFormer checkpoint}; mkdir -p feats results
python3 analysis/extract_features.py --Dataset_name PRID --model_path "$B" --out feats/prid_base.pt
python3 analysis/extract_features.py --Dataset_name PRID --model_path "$X" --out feats/prid_full.pt
python3 analysis/camera_invariance.py --feats feats/prid_base.pt feats/prid_full.pt --names "w/o XCam" "XCamFormer" --out results/inv_prid
python3 analysis/layer_analysis.py    --feats feats/prid_base.pt feats/prid_full.pt --names "w/o XCam" "XCamFormer" --out results/layers_prid
python3 analysis/tsne_identity.py --models base="$B" full="$X" --out results/tsne_blocks_prid
