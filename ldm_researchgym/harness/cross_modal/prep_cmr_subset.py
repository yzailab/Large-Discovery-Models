#!/usr/bin/env python
"""COCO-C 数据准备：清洁软链 + 腐蚀图生成 + 子集标注 + 配置 yaml。

生成的东西（相对任务根目录）：
  dataset/coco-IP/coco_image/val2014            -> 清洁图软链
  dataset/coco-IP/COCO_IP_<corr>_5/val2014/*.jpg 腐蚀图（全量 5000）
  dataset/coco_sub<N>/coco_annotations/coco_karpathy_test.json  子集标注
  configs/QS/i2t/coco_image_c_sub/*.yaml         子集配置（评测内循环用）
  configs/QS/i2t/coco_image_c_full/*.yaml        全量配置（最终复核用）
"""
import json, os, random
from functools import partial
from multiprocessing import Pool

import numpy as np
from PIL import Image
from imagecorruptions import corrupt

ROOT = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(ROOT, "dataset/coco/images/val2014")
ANN = os.path.join(ROOT, "dataset/coco/coco_annotations/coco_karpathy_test.json")
CORRUPTIONS = ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"]
SEVERITY = 5
SUBSET_N = 1000
SEED = 0


def one_image(args, corruption, out_dir):
    name = args
    dst = os.path.join(out_dir, name)
    if os.path.exists(dst):
        return 0
    im = np.array(Image.open(os.path.join(CLEAN, name)).convert("RGB"))
    out = corrupt(im, corruption_name=corruption, severity=SEVERITY)
    Image.fromarray(out).save(dst, quality=95)
    return 1


def main():
    ann = json.load(open(ANN))
    names = sorted({a["image"].split("/")[-1] for a in ann})
    print(f"标注 {len(ann)} 条，图片 {len(names)} 张")

    # 1) 清洁软链
    clean_link = os.path.join(ROOT, "dataset/coco-IP/coco_image")
    os.makedirs(clean_link, exist_ok=True)
    link = os.path.join(clean_link, "val2014")
    if not os.path.exists(link):
        os.symlink(CLEAN, link)
    print("清洁软链 ok")

    # 2) 腐蚀图
    for c in CORRUPTIONS:
        out_dir = os.path.join(ROOT, f"dataset/coco-IP/COCO_IP_{c}_{SEVERITY}/val2014")
        os.makedirs(out_dir, exist_ok=True)
        with Pool(48) as p:
            made = sum(p.map(partial(one_image, corruption=c, out_dir=out_dir), names, chunksize=16))
        print(f"{c}: 新生成 {made} / {len(names)}")

    # 3) 子集标注
    rng = random.Random(SEED)
    keep = set(rng.sample(range(len(ann)), SUBSET_N))
    sub = [a for i, a in enumerate(ann) if i in keep]
    sub_dir = os.path.join(ROOT, f"dataset/coco_sub{SUBSET_N}/coco_annotations")
    os.makedirs(sub_dir, exist_ok=True)
    json.dump(sub, open(os.path.join(sub_dir, "coco_karpathy_test.json"), "w"))
    ncap = sum(len(a["caption"][:5]) for a in sub)
    print(f"子集 {len(sub)} 图 / {ncap} 句 -> {sub_dir}")

    # 4) 配置
    base = open(os.path.join(ROOT, "configs/QS/i2t/coco_image_c/retrieval_coco.yaml")).read()
    for tag, ann_root in [("sub", f"./dataset/coco_sub{SUBSET_N}/coco_annotations/"),
                          ("full", "./dataset/coco/coco_annotations/")]:
        cfg_dir = os.path.join(ROOT, f"configs/QS/i2t/coco_image_c_{tag}")
        os.makedirs(cfg_dir, exist_ok=True)
        for c in CORRUPTIONS + ["clean"]:
            img_root = ("./dataset/coco-IP/coco_image" if c == "clean"
                        else f"./dataset/coco-IP/COCO_IP_{c}_{SEVERITY}")
            txt = base.replace("./dataset/coco-IP/coco_image", img_root)
            txt = txt.replace("./dataset/coco/coco_annotations/", ann_root)
            name = "clean.yaml" if c == "clean" else f"COCO_IP_{c}_{SEVERITY}.yaml"
            open(os.path.join(cfg_dir, name), "w").write(txt)
        print(f"配置写好 {cfg_dir}")


if __name__ == "__main__":
    main()
