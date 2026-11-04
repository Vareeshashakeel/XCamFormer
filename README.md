# XCamFormer: Camera-Metadata-Free Video Person Re-Identification via Cross-Camera Supervised Transformer Learning

Official code for the paper **"XCamFormer: Camera-Metadata-Free Video Person Re-Identification via Cross-Camera Supervised Transformer Learning"** (V. Shakeel, A. Fatima, N. Ahmed, A. N. Abbasi, M. Saeed).

XCamFormer builds on VID-Trans-ReID and removes its camera embedding, so **no camera ID is needed at inference**. Camera labels are used **only in the training loss**: a camera-conditioned supervised contrastive objective gives full positive weight to same-identity pairs from different cameras and a reduced weight (γ) to same-camera pairs. It is applied at:

- intermediate transformer blocks 5 and 8, zero-based, i.e. the 6th and 9th blocks (ICCS);
- the four local part features (PLCCS).

It adds no parameters or computation at inference.

## Repository structure

```text
VID_Trans_ReID.py        training entry point
VID_Trans_model.py       XCamFormer model (ViT-B/16 backbone, global + 4 part branches)
VID_Test.py              evaluation used during training
VID_Test_Final.py        final evaluation (optional --flip_test)
Dataloader.py            data loading and cross-camera identity sampler (P identities x K=4 tracklets)
Datasets/                MARS, iLIDS-VID, PRID2011 loaders
loss/                    ID, triplet, center and cross-camera supervised contrastive losses
analysis/                camera probe, layer-wise analysis, t-SNE, embedding cost, seed summary
scripts/                 one script per experiment in the paper (see below)
splits/                  how to obtain the iLIDS-VID split used for the reported results
```

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3, PyTorch 2.1 (CUDA 11.8).

## Data

All three benchmarks are public. Download them from their official sources and keep the standard folder layout.

| Dataset | Train IDs / tracklets | Query | Gallery |
|---|---|---|---|
| MARS | 625 / 8,298 | 1,980 | 9,330 |
| iLIDS-VID | 150 / 300 | 150 | 150 |
| PRID2011 | 89 / 178 | 89 | 89 |

- **MARS:** use the official train/query/gallery partition.
- **PRID2011:** uses `splits_prid2011.json` (standard splits), split 0.
- **iLIDS-VID:** the reported results use a fixed train/test split of 150/150 identities. Recreate it exactly from the released iLIDS-VID checkpoint with `bash scripts/make_ilids_split.sh iLIDSVID_partxcam_best.pth`, which writes `$ILIDS_ROOT/splits.json` (see `splits/README.md`). Without this file, the loader creates the standard splits from `train_test_splits_ilidsvid.mat`.

Set the paths once:

```bash
export MARS_ROOT=/path/to/MARS
export ILIDS_ROOT=/path/to/ILID
export PRID_ROOT=/path/to/prid
export VIT=/path/to/jx_vit_base_p16_224-80ecf9dd.pth   # ImageNet-pretrained ViT-B/16
```

The ImageNet ViT-B/16 weights (`jx_vit_base_p16_224-80ecf9dd.pth`) are the standard timm release used by TransReID and VID-Trans-ReID.

## Training settings (Table 3 of the paper)

| Dataset | Epochs | Batch | Frames | Eval. every | λ_x | λ_p | γ_x / γ_p | τ | Blocks |
|---|---|---|---|---|---|---|---|---|---|
| MARS | 120 | 64 | 4 | 10 | 0.15 | 0.10 | 0.25 / 0.10 | 0.07 | 5, 8 |
| iLIDS-VID | 160 | 16 | 4 | 20 | 0.10 | 0.08 | 0.25 / 0.10 | 0.07 | 5, 8 |
| PRID2011 | 160 | 16 | 4 | 20 | 0.10 | 0.08 | 0.25 / 0.10 | 0.07 | 5, 8 |

The center-loss weight is 0.0005 and the attention-loss weight is 1.0 for all datasets. The default random seed is 1234.

## Reproducing the paper

| Paper result | Command |
|---|---|
| XCamFormer, MARS (Tables 4, 7) | `bash scripts/train_mars.sh` |
| XCamFormer, iLIDS-VID (Tables 5, 7) | `bash scripts/train_ilidsvid.sh` |
| XCamFormer, PRID2011 (Tables 6, 7) | `bash scripts/train_prid.sh` |
| Camera-removed baseline (E2, "w/o XCam") | `bash scripts/train_baselines.sh Mars` (or `iLIDSVID`, `PRID`) |
| Evaluation; flip-test row of Table 4 | `bash scripts/evaluate.sh Mars <ckpt> [--flip_test]` |
| Run-to-run variation (Table 8) | `bash scripts/prid_seeds.sh` |
| Camera probe, layer analysis, t-SNE (Table 9, Figs. 7–8) | `bash scripts/analysis_prid.sh <baseline.pth> <xcamformer.pth>` |
| Cross-dataset and embedding cost (Tables 10, 12) | `bash scripts/cross_dataset_and_cost.sh <Mars_ckpt>` |

Each script contains the exact command line used in the paper.

## Results (final-epoch checkpoints, no test-time augmentation)

| Dataset | Rank-1 (%) | Rank-5 (%) | mAP (%) |
|---|---|---|---|
| MARS | 89.0 | 96.9 | 85.1 |
| MARS + flip test | 89.4 | 96.8 | 86.4 |
| iLIDS-VID (single split) | 89.3 | 100.0 | 94.1 |
| PRID2011 (single split) | 96.6 | 98.9 | 97.6 |

The camera-removed baseline on MARS reaches 88.4 / 84.7 (Rank-1 / mAP). The results on the small datasets vary noticeably between training runs; see Table 8 of the paper.

## Trained models

| Model | Download |
|---|---|
| XCamFormer – MARS (`Mars_partxcam_best.pth`) | [Google Drive](https://drive.google.com/file/d/1q5sEuF7kD7ZNM50WZ7JrCtuHbEx1qu0i/view?usp=sharing) |
| XCamFormer – iLIDS-VID (`iLIDSVID_partxcam_best.pth`) | [Google Drive](https://drive.google.com/file/d/190bfGlKsxFP7ECnfNxZYNSByqg6g5DBc/view?usp=sharing) |
| XCamFormer – PRID2011 (`PRID_partxcam_best.pth`) | [Google Drive](https://drive.google.com/file/d/1shUENdBlHcvFTOVP1xDpVD08RDMD6ugo/view?usp=sharing) |

These are the checkpoints that produced the results reported in the paper. They are the best checkpoints saved during training. For MARS and PRID2011 the best checkpoint gives the same numbers as the final epoch. For iLIDS-VID it gives 90.0 / 100.0 / 94.4 (Rank-1 / Rank-5 / mAP), which the paper reports as the maximum interim evaluation, next to the final-epoch result of 89.3 / 100.0 / 94.1.

To evaluate a downloaded model:

```bash
bash scripts/evaluate.sh Mars Mars_partxcam_best.pth
```

## Acknowledgement

This code is built on [VID-Trans-ReID](https://github.com/AishahAADU/VID-Trans-ReID) (Alsehaim and Breckon, BMVC 2022).
