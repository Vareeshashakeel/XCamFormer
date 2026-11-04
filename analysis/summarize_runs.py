"""
Aggregate *_results.json files from VID_Trans_ReID.py into mean ± std tables
(10 splits on PRID/iLIDS-VID, several seeds on MARS, ablations) - reviewer R2-15.

Usage
-----
python analysis/summarize_runs.py \
   --group "XCamFormer=runs/prid/full/*_results.json" \
   --group "w/o XCam=runs/prid/base/*_results.json" --out results/prid_10splits.md
"""
import argparse
import glob
import json

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group', action='append', required=True, help='"Label=glob"')
    p.add_argument('--key', default='test_at_selected_checkpoint',
                   choices=['test_at_selected_checkpoint', 'test_at_last_epoch'])
    p.add_argument('--out', default=None)
    a = p.parse_args()

    lines = [f'Metric source: `{a.key}`', '',
             '| Setting | n runs | Rank-1 | Rank-5 | mAP |', '|---|---|---|---|---|']
    for g in a.group:
        label, pattern = g.split('=', 1)
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            lines.append(f'| {label} | 0 | - | - | - |')
            continue
        vals = {m: [] for m in ('rank1', 'rank5', 'mAP')}
        for f in files:
            r = json.load(open(f))[a.key]
            for m in vals:
                vals[m].append(100 * r[m])
        fmt = lambda v: f'{np.mean(v):.1f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.1f}'
        lines.append(f"| {label} | {len(files)} | {fmt(vals['rank1'])} | {fmt(vals['rank5'])} | {fmt(vals['mAP'])} |")
    text = '\n'.join(lines)
    print(text)
    if a.out:
        open(a.out, 'w').write(text + '\n')


if __name__ == '__main__':
    main()
