"""Everything tomorrow's report needs, in one JSON."""
import json, glob, os, re

# 路径全部从环境变量来：RG_ROOT 指 ResearchGym checkout，RG_RUNS 可单独覆盖 runs 目录，
# LDM_REPORT_DIR 放中间产物（基线 grade 的 json、参照跑的日志）。
RG_ROOT = os.environ.get("RG_ROOT", os.path.expanduser("~/ResearchGym"))
RUNS = os.environ.get("RG_RUNS", os.path.join(RG_ROOT, "runs"))
REPORT_DIR = os.environ.get("LDM_REPORT_DIR", "/tmp")
OUT = os.environ.get("LDM_REPORT_JSON", os.path.join(REPORT_DIR, "ldm_report.json"))


def curve(path):
    """Per-evaluation score and running best, in evaluation order."""
    rows, best = [], None
    for line in open(os.path.join(path, "events.jsonl")):
        e = json.loads(line)
        if e["event_type"] != "candidate_evaluated":
            continue
        ev = e["payload"].get("evaluation") or {}
        v = (ev.get("metrics") or {}).get("official_score")
        ok = ev.get("status") == "succeeded" and isinstance(v, (int, float))
        if ok:
            best = v if best is None else max(best, v)
        rows.append({"round": e.get("iteration"), "candidate": e.get("candidate_id"),
                     "score": v if ok else None, "best": best, "status": ev.get("status")})
    return rows


def empty_rounds(path):
    n = 0
    for line in open(os.path.join(path, "events.jsonl")):
        e = json.loads(line)
        if e["event_type"] == "reservoir_built" and not (e["payload"].get("candidate_ids") or []):
            n += 1
    return n


report = {"campaigns": [], "cmr_sub": {}, "cmr_full": {}, "tse_baselines": {}, "cl": {}}

for d in sorted(glob.glob(f"{RUNS}/*/*/ldm_campaign/*")):
    sp = os.path.join(d, "summary.json")
    if not os.path.exists(os.path.join(d, "events.jsonl")):
        continue
    try:
        # 还在跑的 campaign 没有 summary.json，但 events.jsonl 已经有分数了
        s = json.load(open(sp)) if os.path.exists(sp) else {"stop_reason": "running"}
        rows = curve(d)
    except Exception as exc:
        report["campaigns"].append({"path": os.path.relpath(d, RUNS), "error": str(exc)})
        continue
    report["campaigns"].append({
        "path": os.path.relpath(d, RUNS),
        "task": (s.get("task") or "").replace("researchgym_", ""),
        "stop_reason": s.get("stop_reason"),
        "rounds_run": s.get("rounds_run"),
        "successful": s.get("successful_evaluation_count"),
        "failed": s.get("failed_evaluation_count"),
        "empty_rounds": empty_rounds(d),
        "best": max([r["score"] for r in rows if r["score"] is not None], default=None),
        "curve": rows,
    })

# CMR：子协议（1000 图，内循环用）和全量协议（5000 图，论文可比）
for tag, dest in (("cmr_%s_summary.json", "cmr_sub"), ("cmr_full_%s_summary.json", "cmr_full")):
    for f in sorted(glob.glob(os.path.join(REPORT_DIR, tag.replace("%s", "*")))):
        name = os.path.basename(f)
        m = re.match(r"cmr_(?:full_)?(.+)_summary\.json$", name)
        if not m:
            continue
        method = m.group(1)
        if dest == "cmr_sub" and name.startswith("cmr_full_"):
            continue
        if dest == "cmr_sub" and method == "best_full":
            # 2026-09-12 那次全量复核留下的文件，别混进子协议表
            dest_key, method = "cmr_full", "ldm_2026-09-12"
        else:
            dest_key = dest
        s = json.load(open(f))
        for r in s.get("rows") or []:
            report[dest_key][method] = {
                "row": r["row"], "average": r["average"],
                "columns": {c["label"]: c["value"] for c in r["columns"]},
            }
        if method not in report[dest_key]:
            report[dest_key][method] = {"average": None, "missing": s.get("missing")}


def scalarize_tse(summary):
    cells = [stats["mean"]
             for section in ("real", "synthetic")
             for entries in (summary.get(section) or {}).values() if isinstance(entries, dict)
             for stats in entries.values()
             if isinstance(stats, dict) and isinstance(stats.get("mean"), (int, float))]
    return sum(cells) / len(cells) if cells else None


for name in ("IG", "GradSHAP"):
    p = os.path.join(REPORT_DIR, f"grade_{name}.json")
    if os.path.exists(p):
        report["tse_baselines"][name] = scalarize_tse(json.load(open(p)))

# CL：最终复核日志里的三个种子 + SimpleCIL 参照
NUM = r"[0-9]+(?:\.[0-9]+)?"


def cl_log_numbers(paths):
    finals, avgs, forgets = [], [], []
    for f in paths:
        t = open(f, errors="ignore").read()
        curves = re.findall(r"CNN top1 curve: \[([^\]]*)\]", t)
        if curves:
            finals.append([float(x) for x in re.findall(NUM, curves[-1])][-1])
        a = re.findall(r"Average Accuracy \(CNN\): (" + NUM + ")", t)
        if a:
            avgs.append(float(a[-1]))
        g = re.findall(r"Forgetting \(CNN\): (" + NUM + ")", t)
        if g:
            forgets.append(float(g[-1]))
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    return {"final_acc": mean(finals), "avg_acc": mean(avgs), "forgetting": mean(forgets),
            "n": len(finals), "finals": finals}


CL_TASK = os.path.join(RG_ROOT, "tasks/test/continual-learning")
report["cl"]["ldm_v1_final"] = cl_log_numbers(sorted(glob.glob(
    os.path.join(CL_TASK, "logs_final/ldm/cifar224/0/10/final_*.log"))))
report["cl"]["ldm_v2_final"] = cl_log_numbers(sorted(glob.glob(
    os.path.join(CL_TASK, "logs_final_v2/ldm/cifar224/0/10/final2_*.log"))))
report["cl"]["simplecil_ref"] = cl_log_numbers(
    [os.path.join(REPORT_DIR, "cl_ref.log")])

json.dump(report, open(OUT, "w"), indent=2)
print("wrote", OUT)
for c in report["campaigns"]:
    print("  %-46s stop=%-22s ok=%-3s empty=%-2s best=%s" % (
        c.get("path", "?"), c.get("stop_reason"), c.get("successful"),
        c.get("empty_rounds"), c.get("best")))
print("  cmr_sub :", {k: v.get("average") for k, v in report["cmr_sub"].items()})
print("  cmr_full:", {k: v.get("average") for k, v in report["cmr_full"].items()})
print("  tse_base:", report["tse_baselines"])
print("  cl      :", {k: {kk: vv for kk, vv in v.items() if kk != "finals"} for k, v in report["cl"].items()})
