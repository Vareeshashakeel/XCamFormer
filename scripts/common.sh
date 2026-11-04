#!/bin/bash
# Dataset and weight paths. Export them in your shell before running a script, for example:
#   export MARS_ROOT=/data/MARS ILIDS_ROOT=/data/ILID PRID_ROOT=/data/prid
#   export VIT=/weights/jx_vit_base_p16_224-80ecf9dd.pth   # ImageNet ViT-B/16 (needed for training only)
export MARS_ROOT ILIDS_ROOT PRID_ROOT VIT
cd "$(dirname "$0")/.."
