# iLIDS-VID split

The iLIDS-VID results in the paper use one fixed identity-disjoint split (150 training / 150 test identities; split (b) in Sec. 4.7).
It is recovered exactly from the released checkpoint `iLIDSVID_partxcam_best.pth`: each of the 150 outputs of the
identity classifier corresponds to one training identity, and a one-to-one (Hungarian) assignment between classifier
outputs and identities identifies the training set; the remaining identities form the test set.

    export ILIDS_ROOT=/path/to/ILID
    bash scripts/make_ilids_split.sh /path/to/iLIDSVID_partxcam_best.pth

This writes `$ILIDS_ROOT/splits.json` (an existing file is kept as `splits_backup.json`).
Evaluating the released checkpoint on this split gives Rank-1 90.0 %, Rank-5 100.0 %, mAP 94.5 % (best checkpoint).
Identities with a tracklet in only one camera are excluded, as in the loader.
