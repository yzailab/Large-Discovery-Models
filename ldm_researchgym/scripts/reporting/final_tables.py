"""所有结果的绝对值表。缺的部分标 pending。"""
import json, glob, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RG_ROOT = os.environ.get("RG_ROOT", os.path.expanduser("~/ResearchGym"))
RUNS = os.environ.get("RG_RUNS", os.path.join(RG_ROOT, "runs"))
REPORT_JSON = os.environ.get(
    "LDM_REPORT_JSON",
    os.path.join(os.environ.get("LDM_REPORT_DIR", "/tmp"), "ldm_report.json"))

# 先让同目录的 report_data.py 把原始数据收成一份 JSON，再在这里排版。
subprocess.run([sys.executable, str(HERE / "report_data.py")], capture_output=True)
R = json.load(open(REPORT_JSON))
CORR = ["Gauss.", "Shot", "Impul.", "Speckle", "Defoc.", "Glass", "Motion", "Zoom",
        "Snow", "Frost", "Fog", "Brit.", "Contr.", "Elastic", "Pixel", "JPEG"]


def latest(task, day=None):
    best = None
    for c in R["campaigns"]:
        if c.get("task") != task and task not in c["path"]:
            continue
        if day and day not in c["path"]:
            continue
        if best is None or c["path"] > best["path"]:
            best = c
    return best


def cand_cols(run_glob, cid):
    p = next(iter(glob.glob(f"{run_glob}/evaluations/{cid}/grade_summary.json")), None)
    if not p:
        return None
    g = json.load(open(p))
    if not g.get("rows"):
        return None
    return {c["label"]: c["value"] for c in g["rows"][0]["columns"]}, g["rows"][0]["average"]


print("=" * 112)
print("表1  cross-modal-retrieval | 子协议 1000图/5000句, i2t, COCO-C severity 5, BLIP ViT-B/16 | 指标 txt_r1 (Recall@1, 越高越好)")
print("=" * 112)
hdr = "%-22s " % "方法" + " ".join("%7s" % c for c in CORR) + " |   avg"
print(hdr)
rows = []
for name, label in (("source_ref", "source (不适配)"), ("tent", "Tent"), ("sar", "SAR"),
                    ("read", "READ"), ("shot", "SHOT")):
    d = R["cmr_sub"].get(name)
    if d and d.get("columns"):
        rows.append((label, d["columns"], d["average"]))
    else:
        rows.append((label, None, None))
run13 = latest("cross-modal-retrieval", "2026-09-13")
if run13:
    ok = [x for x in run13["curve"] if x["score"] is not None]
    if ok:
        bid = max(ok, key=lambda x: x["score"])["candidate"]
        got = cand_cols(os.path.join(RUNS, run13["path"]), bid)
        if got:
            rows.append(("LDM 最优候选", got[0], got[1]))
for label, cols, avg in rows:
    if cols:
        print("%-22s " % label + " ".join("%7.1f" % cols[c] for c in CORR) + " | %6.2f" % avg)
    else:
        print("%-22s %s" % (label, "pending"))

print()
print("=" * 112)
print("表2  cross-modal-retrieval | 全量协议 5000图/25010句 (论文可比) | 指标 txt_r1")
print("=" * 112)
print("%-22s %s |   avg" % ("方法", " ".join("%7s" % c for c in CORR)))
for name, label in (("source_ref", "source (不适配)"), ("tent", "Tent"),
                    ("ldm", "LDM 最优候选"), ("ldm_2026-09-12", "LDM (09-12 旧场次)")):
    d = R["cmr_full"].get(name)
    if d and d.get("columns"):
        print("%-22s " % label + " ".join("%7.1f" % d["columns"][c] for c in CORR) + " | %6.2f" % d["average"])
    elif name != "ldm_2026-09-12":
        print("%-22s %s" % (label, "pending"))

print()
print("=" * 112)
print("表3  time-series-explanation | 5数据集 × 2替换策略 = 10格 | official_score = 10格均值 (越高越好)")
print("=" * 112)
tse = latest("time-series-explanation")
print("%-22s %12s" % ("方法", "official_score"))
for k, v in R["tse_baselines"].items():
    print("%-22s %12.4f" % ({"IG": "Integrated Gradients", "GradSHAP": "GradientSHAP"}[k], v))
if tse and tse["best"]:
    print("%-22s %12.4f" % ("LDM 最优候选", tse["best"]))

print()
print("=" * 112)
print("表4  continual-learning | CIFAR-100 分10任务, ViT-B/16 预训练, 无回放, 种子1992-1994")
print("     Acc=最后一个任务后在全部100类上的top-1 | AAA=10次测量的均值 | Forgetting=遗忘度(越低越好)")
print("=" * 112)
print("%-30s %8s %8s %12s" % ("方法", "Acc", "AAA", "Forgetting"))
ref = R["cl"]["simplecil_ref"]
v1 = R["cl"]["ldm_v1_final"]
print("%-30s %8.2f %8.2f %12.2f" % ("SimpleCIL (任务自带参照)", ref["final_acc"], ref["avg_acc"], ref["forgetting"]))
print("%-30s %8.2f %8.2f %12.2f" % ("LDM 第一场 (09-10, 修复前)", v1["final_acc"], v1["avg_acc"], v1["forgetting"]))
v2 = R["cl"].get("ldm_v2_final") or {}
if v2.get("final_acc") is not None:
    print("%-30s %8.2f %8.2f %12.2f" % ("LDM 第二场 (09-13, 修复后)",
                                        v2["final_acc"], v2["avg_acc"], v2["forgetting"]))
else:
    cl2 = latest("continual-learning", "2026-09-13")
    print("%-30s %8s   (搜索期 5-epoch 分数, 官方协议复核 pending)" % (
        "LDM 第二场 (09-13, 修复后)",
        ("%.2f" % cl2["best"]) if cl2 and cl2.get("best") else "pending"))

print()
print("=" * 112)
print("表5  搜索过程 | rounds=外层轮数 successful=产出有效分数的评估 empty=提案全被去重掉的空轮")
print("=" * 112)
print("%-46s %-22s %7s %7s %7s %10s" % ("campaign", "停机原因", "rounds", "success", "empty", "best"))
for c in sorted(R["campaigns"], key=lambda x: x["path"]):
    if c.get("successful") in (None, 0) and c.get("best") is None:
        continue
    print("%-46s %-22s %7s %7s %7s %10s" % (
        c["path"].replace("/ldm_campaign", ""), c.get("stop_reason"), c.get("rounds_run"),
        c.get("successful"), c.get("empty_rounds"),
        ("%.4f" % c["best"]) if c.get("best") is not None else "-"))
