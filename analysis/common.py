"""Shared helpers for the revision analyses (feature extraction, retrieval, CKA)."""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from VID_Test import evaluate as cmc_map  # noqa: E402  (same protocol as the paper)

ALL_BLOCKS = tuple(range(11))  # backbone.blocks[:-1] -> 0..10 (block 11 is the copied b1 branch)


# ----------------------------------------------------------------------------- model
def load_model(ckpt_path, device):
    from VID_Trans_model import VID_Trans
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    num_classes = state['classifier.weight'].shape[0]  # lets a MARS model be tested on PRID/iLIDS
    model = VID_Trans(num_classes=num_classes, pretrainpath=None)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


# ----------------------------------------------------------------------------- occlusion
def occlude(imgs, mode, ratio, gen):
    """
    imgs: [clips, T, 3, H, W] (normalised to [-1, 1]; 0 == mean grey).
    mode 'erase': an independent random rectangle (area = ratio) in every frame.
    mode 'band' : one horizontal band (height = ratio * H) at a random height, fixed for the
                  whole tracklet - mimics a static occluder (car, railing, other pedestrian).
    """
    if mode is None or ratio <= 0:
        return imgs
    imgs = imgs.clone()
    c, t, _, h, w = imgs.shape
    if mode == 'band':
        bh = max(1, int(round(ratio * h)))
        y0 = int(torch.randint(0, h - bh + 1, (1,), generator=gen))
        imgs[..., y0:y0 + bh, :] = 0.0
        return imgs
    if mode == 'erase':
        for ci in range(c):
            for ti in range(t):
                area = ratio * h * w
                aspect = float(torch.empty(1).uniform_(0.3, 3.3, generator=gen))
                eh = min(h, int(round((area * aspect) ** 0.5)))
                ew = min(w, int(round((area / aspect) ** 0.5)))
                y0 = int(torch.randint(0, h - eh + 1, (1,), generator=gen))
                x0 = int(torch.randint(0, w - ew + 1, (1,), generator=gen))
                imgs[ci, ti, :, y0:y0 + eh, x0:x0 + ew] = 0.0
        return imgs
    raise ValueError(mode)


# ----------------------------------------------------------------------------- extraction
@torch.no_grad()
def extract(model, loader, device, flip=False, blocks=ALL_BLOCKS, occ_mode=None, occ_ratio=0.0, seed=0, desc=''):
    """Returns dict of per-tracklet features (clip-averaged) + pids, camids, forward times (s)."""
    gen = torch.Generator().manual_seed(seed)
    out = {'full': [], 'global_bn': [], 'global': [], 'parts_bn': []}
    blk = {b: [] for b in blocks}
    pids, camids, times = [], [], []
    for i, (imgs, b_pids, b_camids, _) in enumerate(loader):
        imgs = imgs.squeeze(0)                      # [clips, T, 3, H, W]
        imgs = occlude(imgs, occ_mode, occ_ratio, gen)
        imgs = imgs.to(device, non_blocking=True)
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = model(imgs, None, return_all=True, out_indices=blocks)
        if flip:
            rf = model(torch.flip(imgs, dims=[-1]), None, return_all=True, out_indices=blocks)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

        def agg(a, b=None):
            x = a if b is None else 0.5 * (a + b)
            return x.float().mean(0).cpu()
        for k in out:
            out[k].append(agg(r[k], rf[k] if flip else None))
        for b in blocks:
            blk[b].append(agg(r['blocks'][b], rf['blocks'][b] if flip else None))
        pids.append(int(b_pids[0]))
        camids.append(int(b_camids.view(-1)[0].item()))
        if (i + 1) % 200 == 0:
            print(f'  [{desc}] {i + 1}/{len(loader)} tracklets')
    res = {k: torch.stack(v) for k, v in out.items()}
    res['blocks'] = {b: torch.stack(v) for b, v in blk.items()}
    res['pids'] = np.asarray(pids)
    res['camids'] = np.asarray(camids)
    res['forward_s'] = np.asarray(times)
    return res


# ----------------------------------------------------------------------------- retrieval
def distmat(qf, gf, metric='euclidean'):
    qf, gf = qf.float(), gf.float()
    if metric == 'cosine':
        qf, gf = F.normalize(qf, dim=1), F.normalize(gf, dim=1)
        return (1 - qf @ gf.t()).numpy()
    d = qf.pow(2).sum(1, keepdim=True) + gf.pow(2).sum(1, keepdim=True).t() - 2 * qf @ gf.t()
    return d.clamp_min(0).numpy()


def retrieval(qf, gf, q, g, metric='euclidean'):
    cmc, mAP = cmc_map(distmat(qf, gf, metric), q['pids'], g['pids'], q['camids'], g['camids'])
    return {'rank1': float(cmc[0]), 'rank5': float(cmc[4]) if len(cmc) > 4 else float('nan'), 'mAP': float(mAP)}


def get_variant(d, name):
    if name.startswith('block'):
        return d['blocks'][int(name[5:])]
    return d[name]


# ----------------------------------------------------------------------------- CKA
def linear_cka(x, y):
    """Linear CKA (Kornblith et al., 2019) between two [n, d] feature matrices."""
    x = x.double() - x.double().mean(0, keepdim=True)
    y = y.double() - y.double().mean(0, keepdim=True)
    hsic = (x.t() @ y).pow(2).sum()
    return float(hsic / ((x.t() @ x).pow(2).sum().sqrt() * (y.t() @ y).pow(2).sum().sqrt()))


def load_feats(path):
    return torch.load(path, map_location='cpu', weights_only=False)
