import random
import numpy as np
from PIL import Image, ImageFile

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from utility import CrossCameraRandomIdentitySampler, RandomIdentitySampler, RandomErasing3
from Datasets.MARS_dataset import Mars
from Datasets.iLDSVID import iLIDSVID
from Datasets.PRID_dataset import PRID

ImageFile.LOAD_TRUNCATED_IMAGES = True

__factory = {
    'Mars': Mars,
    'iLIDSVID': iLIDSVID,
    'PRID': PRID,
}


def train_collate_fn(batch):
    imgs, pids, camids, labels = zip(*batch)
    return (
        torch.stack(imgs, dim=0),
        torch.tensor(pids, dtype=torch.int64),
        torch.tensor(camids, dtype=torch.int64),
        torch.stack(labels, dim=0),
    )


def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, torch.tensor(camids, dtype=torch.int64), img_paths


def build_dataset(Dataset_name, split_id=0):
    """PRID/iLIDS-VID accept a split id (10 standard splits); MARS has a single fixed split."""
    if Dataset_name in ('PRID', 'iLIDSVID'):
        return __factory[Dataset_name](split_id=split_id)
    if split_id != 0:
        print(f'[WARN] {Dataset_name} has a single official split; ignoring split_id={split_id}')
    return __factory[Dataset_name]()


def carve_validation_split(train, num_val_ids, seed=0):
    """
    Hold out `num_val_ids` training identities as a validation set so that checkpoints and
    hyperparameters can be selected without touching the test set (reviewer R2-16).

    Returns (new_train, val_query, val_gallery, num_classes). Remaining training identities are
    relabelled to 0..M-1. For every held-out identity one tracklet is used as query and all its
    other tracklets go to the gallery; only identities seen by >= 2 cameras are held out so that
    each validation query has at least one cross-camera match.
    """
    rng = random.Random(seed)
    by_pid = {}
    for item in train:
        by_pid.setdefault(item[1], []).append(item)
    eligible = sorted(p for p, items in by_pid.items() if len({it[2] for it in items}) >= 2)
    if num_val_ids > len(eligible) // 2:
        raise ValueError(f'num_val_ids={num_val_ids} is too large ({len(eligible)} multi-camera training ids)')
    val_pids = set(rng.sample(eligible, num_val_ids))

    keep_pids = sorted(p for p in by_pid if p not in val_pids)
    relabel = {p: i for i, p in enumerate(keep_pids)}
    new_train = [(paths, relabel[pid], cam) for (paths, pid, cam) in train if pid in relabel]

    val_query, val_gallery = [], []
    for pid in sorted(val_pids):
        items = by_pid[pid][:]
        rng.shuffle(items)
        q = items[0]
        val_query.append(q)
        val_gallery.extend(items[1:])
    return new_train, val_query, val_gallery, len(keep_pids)


def corrupt_camera_labels(train, fraction, seed=0):
    """Reassign the camera label of `fraction` of training tracklets to a different random camera
    (persistent label noise; affects the sampler and the XCam losses, never the input). R3-3 / R2-14."""
    rng = random.Random(seed + 1000)
    cams = sorted({c for _, _, c in train})
    idx = set(rng.sample(range(len(train)), int(round(fraction * len(train)))))
    out = []
    for i, (paths, pid, cam) in enumerate(train):
        if i in idx:
            cam = rng.choice([c for c in cams if c != cam])
        out.append((paths, pid, cam))
    return out, len(idx)


def dataloader(Dataset_name, batch_size=64, num_workers=4, seq_len=4,
               split_id=0, sampler='crosscam', num_instances=4, val_ids=0, val_seed=0, cam_noise=0.0):
    """
    Returns the original 7-tuple. When val_ids > 0, two extra loaders are appended:
    (..., val_q_loader, val_g_loader), built from identities held out of the training set.
    """
    train_transforms = T.Compose([
        T.Resize([256, 128], interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = build_dataset(Dataset_name, split_id=split_id)
    train_list = dataset.train
    num_classes = dataset.num_train_pids
    val_query = val_gallery = None
    if val_ids > 0:
        train_list, val_query, val_gallery, num_classes = carve_validation_split(train_list, val_ids, seed=val_seed)
        print(f'[INFO] Validation split: {val_ids} held-out ids, {len(val_query)} query / {len(val_gallery)} gallery tracklets; '
              f'{num_classes} training ids remain')

    if cam_noise > 0:
        train_list, n_bad = corrupt_camera_labels(train_list, cam_noise, seed=val_seed)
        print(f'[INFO] Camera-label noise: {n_bad}/{len(train_list)} training tracklets reassigned to a wrong camera')

    train_set = VideoDataset_inderase(train_list, seq_len=seq_len, sample='intelligent', transform=train_transforms)
    cam_num = dataset.num_train_cams
    view_num = len(train_list)

    if sampler == 'crosscam':
        train_sampler = CrossCameraRandomIdentitySampler(train_list, batch_size, num_instances)
    elif sampler == 'standard':
        train_sampler = RandomIdentitySampler(train_list, batch_size, num_instances)
    else:
        raise ValueError(f'Unknown sampler: {sampler}')

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=True,
    )

    q_val_set = VideoDataset(dataset.query, seq_len=seq_len, sample='dense', transform=val_transforms)
    g_val_set = VideoDataset(dataset.gallery, seq_len=seq_len, sample='dense', transform=val_transforms)

    q_val_loader = DataLoader(q_val_set, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=val_collate_fn, pin_memory=True)
    g_val_loader = DataLoader(g_val_set, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=val_collate_fn, pin_memory=True)

    out = (train_loader, len(dataset.query), num_classes, cam_num, view_num, q_val_loader, g_val_loader)
    if val_ids > 0:
        vq = DataLoader(VideoDataset(val_query, seq_len=seq_len, sample='dense', transform=val_transforms),
                        batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=val_collate_fn, pin_memory=True)
        vg = DataLoader(VideoDataset(val_gallery, seq_len=seq_len, sample='dense', transform=val_transforms),
                        batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=val_collate_fn, pin_memory=True)
        out = out + (vq, vg)
    return out


def test_loaders(Dataset_name, split_id=0, seq_len=4, num_workers=4, which=('query', 'gallery'), max_items=None):
    """Lightweight helper for analysis scripts: dense-clip loaders for query/gallery/train without building the train sampler."""
    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = build_dataset(Dataset_name, split_id=split_id)
    loaders = {}
    for name in which:
        items = getattr(dataset, name)
        if max_items is not None and len(items) > max_items:
            rng = random.Random(0)
            items = rng.sample(items, max_items)
        loaders[name] = DataLoader(VideoDataset(items, seq_len=seq_len, sample='dense', transform=val_transforms),
                                   batch_size=1, shuffle=False, num_workers=num_workers,
                                   collate_fn=val_collate_fn, pin_memory=True)
    return loaders, dataset


def read_image(img_path):
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def _pad_indices(indices, seq_len):
    indices = list(indices)
    if len(indices) == 0:
        raise RuntimeError('Empty tracklet encountered.')
    while len(indices) < seq_len:
        indices.append(indices[-1])
    return indices


class VideoDataset(Dataset):
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None, max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        if self.sample == 'random':
            frame_indices = list(range(num))
            rand_end = max(0, len(frame_indices) - self.seq_len)
            begin_index = random.randint(0, rand_end)
            indices = frame_indices[begin_index:begin_index + self.seq_len]
            indices = _pad_indices(indices, self.seq_len)
            imgs, targt_cam = [], []
            for idx in indices:
                img = read_image(img_paths[int(idx)])
                if self.transform is not None:
                    img = self.transform(img)
                imgs.append(img.unsqueeze(0))
                targt_cam.append(camid)
            imgs = torch.cat(imgs, dim=0)
            return imgs, pid, targt_cam

        elif self.sample == 'dense':
            cur_index = 0
            frame_indices = list(range(num))
            indices_list = []
            while num - cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len
            last_seq = frame_indices[cur_index:]
            last_seq = _pad_indices(last_seq, self.seq_len)
            indices_list.append(last_seq)

            imgs_list = []
            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break
                imgs = []
                for idx in indices:
                    img = read_image(img_paths[int(idx)])
                    if self.transform is not None:
                        img = self.transform(img)
                    imgs.append(img.unsqueeze(0))
                imgs = torch.cat(imgs, dim=0)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid, img_paths

        elif self.sample == 'dense_subset':
            frame_indices = list(range(num))
            rand_end = max(0, len(frame_indices) - self.max_length - 1)
            begin_index = random.randint(0, rand_end)
            cur_index = begin_index
            indices_list = []
            while num - cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len
            last_seq = frame_indices[cur_index:]
            last_seq = _pad_indices(last_seq, self.seq_len)
            indices_list.append(last_seq)
            imgs_list = []
            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break
                imgs = []
                for idx in indices:
                    img = read_image(img_paths[int(idx)])
                    if self.transform is not None:
                        img = self.transform(img)
                    imgs.append(img.unsqueeze(0))
                imgs = torch.cat(imgs, dim=0)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid

        elif self.sample == 'intelligent_random':
            indices = []
            each = max(num // self.seq_len, 1)
            for i in range(self.seq_len):
                if i != self.seq_len - 1:
                    indices.append(random.randint(min(i * each, num - 1), min((i + 1) * each - 1, num - 1)))
                else:
                    indices.append(random.randint(min(i * each, num - 1), num - 1))
            imgs = []
            for idx in indices:
                img = read_image(img_paths[int(idx)])
                if self.transform is not None:
                    img = self.transform(img)
                imgs.append(img.unsqueeze(0))
            imgs = torch.cat(imgs, dim=0)
            return imgs, pid, camid
        else:
            raise KeyError('Unknown sample method: {}.'.format(self.sample))


class VideoDataset_inderase(Dataset):
    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None, max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length
        self.erase = RandomErasing3(probability=0.5, mean=[0.485, 0.456, 0.406])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        if self.sample != 'intelligent':
            frame_indices = list(range(num))
            rand_end = max(0, len(frame_indices) - self.seq_len)
            begin_index = random.randint(0, rand_end)
            indices = frame_indices[begin_index:begin_index + self.seq_len]
            indices = _pad_indices(indices, self.seq_len)
        else:
            indices = []
            each = max(num // self.seq_len, 1)
            for i in range(self.seq_len):
                if i != self.seq_len - 1:
                    indices.append(random.randint(min(i * each, num - 1), min((i + 1) * each - 1, num - 1)))
                else:
                    indices.append(random.randint(min(i * each, num - 1), num - 1))

        imgs, labels, targt_cam = [], [], []
        for idx in indices:
            img = read_image(img_paths[int(idx)])
            if self.transform is not None:
                img = self.transform(img)
            img, temp = self.erase(img)
            labels.append(temp)
            imgs.append(img.unsqueeze(0))
            targt_cam.append(camid)

        labels = torch.tensor(labels)
        imgs = torch.cat(imgs, dim=0)
        return imgs, pid, targt_cam, labels
