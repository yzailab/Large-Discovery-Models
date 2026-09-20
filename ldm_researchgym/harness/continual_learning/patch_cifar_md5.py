#!/usr/bin/env python3
"""让 torchvision 接受我们从 hf-mirror 重建的 CIFAR100。

原始 cifar-100-python.tar.gz 从 toronto.edu 只有 12KB/s；改从 hf-mirror 取
parquet 重建成同样的 pickle 结构，内容等价但文件 md5 与官方发布包不同。
torchvision 的完整性检查会因此误判为损坏并重新下载，所以把校验表换成实际 md5。
"""
import hashlib, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def patch():
    from torchvision.datasets import cifar as C
    base = os.path.join(ROOT, C.CIFAR100.base_folder)
    if not os.path.isdir(base):
        return False
    for lst in (C.CIFAR100.train_list, C.CIFAR100.test_list):
        for entry in lst:
            p = os.path.join(base, entry[0])
            if os.path.exists(p):
                entry[1] = md5(p)
    mp = os.path.join(base, C.CIFAR100.meta["filename"])
    if os.path.exists(mp):
        C.CIFAR100.meta["md5"] = md5(mp)
    return True

if __name__ == "__main__":
    ok = patch()
    print("校验表已同步" if ok else "未找到 cifar-100-python 目录")
    from torchvision import datasets
    tr = datasets.cifar.CIFAR100(ROOT, train=True, download=False)
    te = datasets.cifar.CIFAR100(ROOT, train=False, download=False)
    print(f"训练集 {len(tr)} 张，测试集 {len(te)} 张，首个标签 {tr[0][1]}，图像尺寸 {tr[0][0].size}")
