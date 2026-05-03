"""
ETT (Electricity Transformer Temperature) dataset loader.

Downloads ETTh1.csv from GitHub if not present.
Slices into overlapping windows (stride < window for data augmentation),
then randomly masks ~50% of time steps to simulate irregular sampling.

Each sample: (t_obs, x_obs, t_query, x_query, mask)
  t_obs    : (S,)       observed time points (subset of window)
  x_obs    : (S, D)     observed values
  t_query  : (L,)       all time points in window
  x_query  : (L, D)     all values (reconstruction target)
  mask_obs : (L,)       bool, True = observed
"""

import os
import urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ETT_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]  # 7 features

ETT_FILES = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
}


def download_ett(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    paths = []
    for name, url in ETT_FILES.items():
        path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(path):
            print(f"Downloading {name}.csv → {path}")
            urllib.request.urlretrieve(url, path)
        paths.append(path)
    return paths


def load_ett_array(data_dir="data"):
    """Load all ETT files, normalize each independently, then concatenate."""
    import csv
    paths = download_ett(data_dir)
    arrays = []
    for path in paths:
        rows = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append([float(row[c]) for c in ETT_COLS])
        arr = np.array(rows, dtype=np.float32)   # (T_i, 7)
        # normalize per file using its own train-split stats (first 70%)
        n_tr = int(0.70 * len(arr))
        mu   = arr[:n_tr].mean(0)
        std  = arr[:n_tr].std(0) + 1e-6
        arrays.append((arr - mu) / std)
    return np.concatenate(arrays, axis=0)         # (T_total, 7)


class ETTIrregularDataset(Dataset):
    """
    Slices ETT into overlapping windows, randomly masks observations.

    Args:
        data_dir : path to data directory
        window   : length of each time window (hours)
        stride   : step between windows; stride < window → overlapping windows
        obs_frac : fraction of time steps revealed as observations
        split    : 'train' | 'val' | 'test'
        seed     : RNG seed for masking
    """

    def __init__(self, data_dir="data", window=96, stride=24,
                 obs_frac=0.5, split="train", seed=42):
        super().__init__()
        raw = load_ett_array(data_dir)          # (T_total, 7), already normalized per file
        T, D = raw.shape

        # build overlapping windows via sliding window
        starts = np.arange(0, T - window + 1, stride)
        windows = np.stack([raw[s: s + window] for s in starts])  # (N, L, D)
        N = len(windows)

        # chronological split: 70/15/15 on window starts
        n_train = int(0.70 * N)
        n_val   = int(0.15 * N)
        if split == 'train':
            self.data = windows[:n_train]
        elif split == 'val':
            self.data = windows[n_train: n_train + n_val]
        else:
            self.data = windows[n_train + n_val:]

        self.window   = window
        self.obs_frac = obs_frac
        self.seed     = seed
        self.D        = D
        self.t_grid   = np.linspace(0, 1, window, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]          # (L, D)
        L = self.window

        rng   = np.random.default_rng(self.seed + idx)
        n_obs = max(2, int(L * self.obs_frac))
        obs_idx = np.sort(rng.choice(L, size=n_obs, replace=False))

        t_obs = self.t_grid[obs_idx]
        x_obs = x[obs_idx]

        mask = np.zeros(L, dtype=bool)
        mask[obs_idx] = True

        return (
            torch.from_numpy(t_obs),
            torch.from_numpy(x_obs),
            torch.from_numpy(self.t_grid),
            torch.from_numpy(x),
            torch.from_numpy(mask),
        )


def collate_fn(batch):
    """Pad variable-length obs sequences to same length within batch."""
    t_obs_list, x_obs_list, t_query, x_query, masks = zip(*batch)

    max_s = max(t.shape[0] for t in t_obs_list)
    D     = x_obs_list[0].shape[-1]

    t_obs_pad = torch.zeros(len(batch), max_s)
    x_obs_pad = torch.zeros(len(batch), max_s, D)
    obs_mask  = torch.zeros(len(batch), max_s, dtype=torch.bool)

    for i, (t, x) in enumerate(zip(t_obs_list, x_obs_list)):
        s = t.shape[0]
        t_obs_pad[i, :s] = t
        x_obs_pad[i, :s] = x
        obs_mask[i,  :s] = True

    return (
        t_obs_pad,
        x_obs_pad,
        obs_mask,
        torch.stack(list(t_query)),
        torch.stack(list(x_query)),
        torch.stack(list(masks)),
    )


def make_loaders(data_dir="data", window=96, stride=24, obs_frac=0.5,
                 batch_size=64, num_workers=0, seed=42):
    kwargs = dict(data_dir=data_dir, window=window, stride=stride,
                  obs_frac=obs_frac, seed=seed)
    train_ds = ETTIrregularDataset(split='train', **kwargs)
    val_ds   = ETTIrregularDataset(split='val',   **kwargs)
    test_ds  = ETTIrregularDataset(split='test',  **kwargs)

    pin = torch.cuda.is_available()
    loader_kw = dict(batch_size=batch_size, collate_fn=collate_fn,
                     num_workers=num_workers, pin_memory=pin)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    print(f"ETT splits — train:{len(train_ds)}  val:{len(val_ds)}  test:{len(test_ds)}  "
          f"(window={window}, stride={stride})")
    return train_loader, val_loader, test_loader
