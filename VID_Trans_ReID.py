import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.cuda import amp

try:
    from torch_ema import ExponentialMovingAverage
except Exception:
    ExponentialMovingAverage = None

from Dataloader import dataloader
from Loss_fun import make_loss
from VID_Test import test
from VID_Trans_model import VID_Trans
from loss.xcamera_supcon import CrossCameraSupConLoss
from utility import AverageMeter, optimizer, scheduler


def set_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_block_indices(text):
    if isinstance(text, (list, tuple)):
        return tuple(int(x) for x in text)
    return tuple(sorted({int(x.strip()) for x in str(text).split(',') if x.strip() != ''}))


def evaluate_full(model, q_loader, g_loader):
    cmc, mAP = test(model, q_loader, g_loader, return_cmc=True)
    r = lambda k: float(cmc[k - 1]) if len(cmc) >= k else float('nan')
    return {'rank1': r(1), 'rank5': r(5), 'rank10': r(10), 'rank20': r(20), 'mAP': float(mAP)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='XCamFormer: VID-Trans-ReID without camera embedding + intermediate/part-level cross-camera SupCon'
    )
    parser.add_argument('--Dataset_name', required=True, type=str)
    parser.add_argument('--model_path', required=True, type=str, help='ViT pretrained weight path')
    parser.add_argument('--output_dir', default='./output_xcam_intermediate', type=str)
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--eval_every', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seq_len', default=4, type=int)
    parser.add_argument('--center_w', default=0.0005, type=float)
    parser.add_argument('--attn_w', default=1.0, type=float)
    # Intermediate Cross-Camera Supervision (ICCS). --xcam_w 0 disables it.
    parser.add_argument('--xcam_w', default=0.15, type=float)
    parser.add_argument('--xcam_temp', default=0.07, type=float)
    parser.add_argument('--xcam_same_cam_w', default=0.25, type=float,
                        help='gamma for ICCS: 0 = cross-camera positives only, 1 = standard (uniform) SupCon')
    # Part-Level Cross-Camera Supervision (PLCCS). --part_xcam_w 0 disables it.
    parser.add_argument('--part_xcam_w', default=0.10, type=float)
    parser.add_argument('--part_xcam_temp', default=0.07, type=float)
    parser.add_argument('--part_xcam_same_cam_w', default=0.10, type=float,
                        help='gamma for PLCCS: 0 = cross-camera positives only, 1 = standard (uniform) SupCon')
    parser.add_argument('--xcam_blocks', default='5,8', type=str, help='0-based block ids from backbone.blocks[:-1]')

    # ---- added for the revision ----
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--split_id', default=0, type=int, help='PRID/iLIDS-VID split (0-9)')
    parser.add_argument('--sampler', default='crosscam', choices=['crosscam', 'standard'],
                        help='crosscam = camera-aware PK sampler (submitted version); standard = plain PK sampler')
    parser.add_argument('--num_instances', default=4, type=int, help='K tracklets per identity in a batch')
    parser.add_argument('--val_ids', default=0, type=int,
                        help='hold out this many TRAINING identities as a validation set; checkpoint selection then '
                             'uses validation only and the test set is evaluated once at the end')
    parser.add_argument('--select_metric', default='rank1', choices=['rank1', 'mAP'])
    parser.add_argument('--sched_epochs', default=120, type=int,
                        help='cosine schedule length (submitted version used 120 for every dataset)')
    parser.add_argument('--cam_noise', default=0.0, type=float,
                        help='fraction of training tracklets whose camera label is randomly reassigned (robustness test)')
    parser.add_argument('--run_name', default=None, type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    xcam_blocks = parse_block_indices(args.xcam_blocks)
    run_name = args.run_name or f'{args.Dataset_name}_s{args.split_id}_seed{args.seed}'

    loaders = dataloader(
        args.Dataset_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seq_len=args.seq_len,
        split_id=args.split_id,
        sampler=args.sampler,
        num_instances=args.num_instances,
        val_ids=args.val_ids,
        val_seed=args.seed,
        cam_noise=args.cam_noise,
    )
    train_loader, _, num_classes, camera_num, view_num, q_test_loader, g_test_loader = loaders[:7]
    if args.val_ids > 0:
        q_sel_loader, g_sel_loader = loaders[7], loaders[8]
        selection_set = 'validation'
    else:
        q_sel_loader, g_sel_loader = q_test_loader, g_test_loader
        selection_set = 'test'

    model = VID_Trans(
        num_classes=num_classes,
        camera_num=camera_num,
        pretrainpath=args.model_path,
        xcam_block_indices=xcam_blocks,
    )

    loss_fun, center_criterion = make_loss(num_classes=num_classes)
    xcam_criterion = CrossCameraSupConLoss(temperature=args.xcam_temp, same_cam_weight=args.xcam_same_cam_w)
    part_xcam_criterion = CrossCameraSupConLoss(temperature=args.part_xcam_temp, same_cam_weight=args.part_xcam_same_cam_w)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    center_criterion = center_criterion.to(device)

    optimizer_main = optimizer(model)
    lr_scheduler = scheduler(optimizer_main, num_epochs=args.sched_epochs)
    scaler = amp.GradScaler(enabled=(device == 'cuda'))

    optimizer_center = None
    if args.center_w > 0:
        optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.5)

    loss_meter = AverageMeter()
    id_meter = AverageMeter()
    xcam_meter = AverageMeter()
    part_xcam_meter = AverageMeter()
    acc_meter = AverageMeter()
    # NOTE: EMA weights are tracked but NOT used for evaluation (same as the submitted version).
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995) if ExponentialMovingAverage is not None else None
    best_score = -1.0
    best_epoch = None
    history = []
    best_path = os.path.join(args.output_dir, f'{run_name}_best.pth')
    latest_path = os.path.join(args.output_dir, f'{run_name}_latest.pth')

    print(f'[INFO] Device: {device} | run: {run_name} | selection on: {selection_set}')
    print(f'[INFO] XCam blocks: {xcam_blocks} | sampler: {args.sampler} (K={args.num_instances})')
    print(f'[INFO] Loss weights -> center: {args.center_w}, attn: {args.attn_w}, '
          f'xcam: {args.xcam_w} (gamma {args.xcam_same_cam_w}), part_xcam: {args.part_xcam_w} (gamma {args.part_xcam_same_cam_w})')

    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss_meter.reset(); id_meter.reset(); xcam_meter.reset(); part_xcam_meter.reset(); acc_meter.reset()
        lr_scheduler.step(epoch)
        model.train()

        for iteration, (img, pid, target_cam, labels2) in enumerate(train_loader, start=1):
            optimizer_main.zero_grad()
            if optimizer_center is not None:
                optimizer_center.zero_grad()

            img = img.to(device)
            pid = pid.to(device)
            target_cam = target_cam.to(device)
            seq_cam = target_cam[:, 0].contiguous() if target_cam.ndim > 1 else target_cam.contiguous()
            labels2 = labels2.to(device)

            with amp.autocast(enabled=(device == 'cuda')):
                score, feat, a_vals, aux = model(img, pid, cam_label=seq_cam)

                attn_noise = a_vals * labels2
                attn_loss = attn_noise.sum(1).mean()

                loss_id, center = loss_fun(score, feat, pid, seq_cam)
                loss = loss_id + args.center_w * center + args.attn_w * attn_loss

                # Skip the contrastive terms entirely when their weight is 0 (clean ablations).
                if args.xcam_w > 0:
                    xcam_loss = xcam_criterion(aux['xcam_feats'], pid, seq_cam)
                    loss = loss + args.xcam_w * xcam_loss
                else:
                    xcam_loss = torch.zeros((), device=device)
                if args.part_xcam_w > 0:
                    part_xcam_loss = part_xcam_criterion(aux['part_xcam_feats'], pid, seq_cam)
                    loss = loss + args.part_xcam_w * part_xcam_loss
                else:
                    part_xcam_loss = torch.zeros((), device=device)

            scaler.scale(loss).backward()
            scaler.step(optimizer_main)

            if optimizer_center is not None:
                for param in center_criterion.parameters():
                    if param.grad is not None:
                        param.grad.data *= (1.0 / args.center_w)
                scaler.step(optimizer_center)

            scaler.update()
            if ema is not None:
                ema.update()

            acc = (score[0].max(1)[1] == pid).float().mean() if isinstance(score, list) else (score.max(1)[1] == pid).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            id_meter.update(loss_id.item(), img.shape[0])
            xcam_meter.update(float(xcam_loss), img.shape[0])
            part_xcam_meter.update(float(part_xcam_loss), img.shape[0])
            acc_meter.update(acc.item(), 1)

            if device == 'cuda':
                torch.cuda.synchronize()

            if iteration % 50 == 0:
                print('Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, ID+Tri: {:.3f}, XCam: {:.3f}, PartXCam: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}'.format(
                    epoch, iteration, len(train_loader), loss_meter.avg, id_meter.avg, xcam_meter.avg,
                    part_xcam_meter.avg, acc_meter.avg, lr_scheduler._get_lr(epoch)[0]))

        print('Epoch {} finished in {:.1f}s'.format(epoch, time.time() - start_time))

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            res = evaluate_full(model, q_sel_loader, g_sel_loader)
            res['epoch'] = epoch
            res['set'] = selection_set
            history.append(res)
            print(f"[{selection_set}] epoch {epoch}: Rank-1 {res['rank1']:.4f}  mAP {res['mAP']:.4f}")

            torch.save(model.state_dict(), latest_path)
            score_now = res[args.select_metric]
            if score_now > best_score:
                best_score = score_now
                best_epoch = epoch
                torch.save(model.state_dict(), best_path)
                print(f'[OK] Saved best checkpoint ({selection_set} {args.select_metric}={score_now:.4f}): {best_path}')

    train_hours = (time.time() - train_start) / 3600.0

    # ---- final test-set evaluation ----
    model.eval()
    test_last = evaluate_full(model, q_test_loader, g_test_loader)
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_best = evaluate_full(model, q_test_loader, g_test_loader)

    summary = {
        'run_name': run_name,
        'config': vars(args),
        'num_train_ids': num_classes,
        'selection_set': selection_set,
        'best_epoch': best_epoch,
        'test_at_selected_checkpoint': test_best,
        'test_at_last_epoch': test_last,
        'history': history,
        'train_hours': train_hours,
    }
    out_json = os.path.join(args.output_dir, f'{run_name}_results.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print('[DONE] Test @ selected checkpoint (epoch {}): R1 {:.4f} R5 {:.4f} mAP {:.4f} | last epoch: R1 {:.4f} mAP {:.4f}'.format(
        best_epoch, test_best['rank1'], test_best['rank5'], test_best['mAP'], test_last['rank1'], test_last['mAP']))
    print(f'[DONE] Results written to {out_json}')
