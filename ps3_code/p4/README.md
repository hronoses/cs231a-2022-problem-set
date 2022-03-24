# Downloading dataset

1. Download dataset.

```
wget -i kitti_dataset.txt -P /path/to/dataset
```

2. Unzip and reorganize dataset as following
   Every scenes else than those in `val` directory below are
   training set.

```
dataset
├── train/
│   ├── 2011_09_26_drive_0001_sync/
│   ├── 2011_09_26_drive_0005_sync/
│   ├──     ...
│   └── 2011_10_03_drive_0047_sync/
└── val/
    ├── 2011_09_26_drive_0002_sync/
    ├── 2011_09_26_drive_0014_sync/
    ├── 2011_09_26_drive_0020_sync/
    ├── 2011_09_26_drive_0079_sync/
    ├── 2011_09_29_drive_0071_sync/
    ├── 2011_09_30_drive_0033_sync/
    └── 2011_10_03_drive_0042_sync/
```

3. Train the network.

```
python train.py
```
