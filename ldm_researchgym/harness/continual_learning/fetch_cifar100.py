#!/usr/bin/env python3
"""从 hf-mirror 拉 CIFAR100 parquet，重建 torchvision 期望的 pickle 目录结构。

torchvision.datasets.CIFAR100 要的是原始 cifar-100-python/{train,test,meta}，
每个是 pickle：{'data': (N,3072) uint8, 'fine_labels': [...], 'coarse_labels': [...]}。
"""
import io, os, pickle, subprocess, sys
import numpy as np

BASE = "https://hf-mirror.com/datasets/uoft-cs/cifar100/resolve/main/cifar100"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cifar-100-python")
TMP = "/tmp/cifar100_pq"
os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)

import pyarrow.parquet as pq
from PIL import Image

def fetch(name, url):
    p = os.path.join(TMP, name)
    if os.path.exists(p) and os.path.getsize(p) > 1_000_000:
        print(f"  {name} 已存在 {os.path.getsize(p)//1048576}MB"); return p
    print(f"  下载 {name} …")
    r = subprocess.run(["curl", "-fsSL", "--retry", "5", "-o", p, url])
    if r.returncode != 0: sys.exit(f"下载失败: {url}")
    print(f"  {name} {os.path.getsize(p)//1048576}MB"); return p

def build(split, parquet_path, out_name):
    t = pq.read_table(parquet_path)
    cols = t.column_names
    print(f"  {split} 列: {cols}  行数 {t.num_rows}")
    imgs = t.column("img").to_pylist()
    fine = t.column("fine_label").to_pylist()
    coarse = t.column("coarse_label").to_pylist()
    data = np.zeros((len(imgs), 3072), dtype=np.uint8)
    for i, rec in enumerate(imgs):
        b = rec["bytes"] if isinstance(rec, dict) else rec
        a = np.array(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8)  # (32,32,3)
        data[i] = a.transpose(2, 0, 1).reshape(-1)                              # C,H,W 扁平
    # torchvision 用 pickle.load(f, encoding="latin1") 读原始 Python 2 pickle，
    # 键会解成 str；所以这里必须用 str 键，不能用 bytes 键。
    obj = {"data": data, "fine_labels": fine, "coarse_labels": coarse,
           "filenames": [f"{split}_{i}.png" for i in range(len(imgs))],
           "batch_label": f"{split} batch"}
    with open(os.path.join(OUT, out_name), "wb") as f:
        pickle.dump(obj, f)
    print(f"  写入 {out_name}: data{data.shape}")

print("=== 下载 ===")
tr = fetch("train.parquet", f"{BASE}/train-00000-of-00001.parquet")
te = fetch("test.parquet",  f"{BASE}/test-00000-of-00001.parquet")
print("=== 重建 pickle ===")
build("train", tr, "train")
build("test",  te, "test")
with open(os.path.join(OUT, "meta"), "wb") as f:
    pickle.dump({"fine_label_names": [f"c{i}" for i in range(100)],
                 "coarse_label_names": [f"s{i}" for i in range(20)]}, f)
print("meta 写入")
print("=== 同步 md5 校验表并校验 ===")
import patch_cifar_md5; patch_cifar_md5.patch()
from torchvision import datasets
d = datasets.cifar.CIFAR100(os.path.join(os.path.dirname(OUT)), train=True, download=False)
print("  训练集", len(d), "样本; 第一个 label =", d[0][1])
d2 = datasets.cifar.CIFAR100(os.path.join(os.path.dirname(OUT)), train=False, download=False)
print("  测试集", len(d2), "样本")
print("CIFAR100 就绪")
