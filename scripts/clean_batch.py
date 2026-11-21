"""数据清洗：去重 batch_log.csv、标记段三分类统计、重试错误段。
用法:
  python clean_batch.py                对 output/ds/ 下全部序列出报告（含去重）
  python clean_batch.py <seq>          单个序列
  python clean_batch.py <seq> --retry  先重试 error 段（调 process_input_batch 补跑），再报告
"""
import os
import sys
import csv
import subprocess
from collections import Counter

OUT = r"C:\Users\84261\Desktop\BroadTrack\output\ds"
PYTHON = r"C:\Users\84261\.conda\envs\soccer\python.exe"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def processed_sequences():
    return sorted(d for d in os.listdir(OUT)
                  if os.path.isfile(os.path.join(OUT, d, "batch_log.csv")))


def dedupe_csv(seq):
    """同段保留最后一行（重试后的旧 error 行会被覆盖）。"""
    p = os.path.join(OUT, seq, "batch_log.csv")
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    seen = {}
    for r in rows:
        seen[r["seg"]] = r
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for seg in sorted(seen):
            w.writerow(seen[seg])
    return list(seen.values())


def classify(rows):
    """标记段三分类：fail(<0.25 检测失败/素材限制)、weak(0.25-0.45)、pos-only(>=0.45 仅位置偏差)。"""
    fl = [r for r in rows if r["status"] == "flagged"]
    low = sum(1 for r in fl if float(r["score_mean"]) < 0.25)
    mid = sum(1 for r in fl if 0.25 <= float(r["score_mean"]) < 0.45)
    hi = len(fl) - low - mid
    return low, mid, hi


def report(seq, rows):
    c = Counter(r["status"] for r in rows)
    ok = c.get("ok", 0)
    fl = c.get("flagged", 0)
    er = c.get("error", 0)
    ok_rows = [r for r in rows if r["status"] == "ok"]
    ok_mean = round(sum(float(r["score_mean"]) for r in ok_rows) / len(ok_rows), 3) if ok_rows else 0
    low, mid, hi = classify(rows)
    errs = ", ".join(r["seg"] for r in rows if r["status"] == "error")
    print(f"{seq:20s} total={len(rows):4d}  ok={ok:4d} ({ok_mean:.3f})  "
          f"flagged={fl:4d} [fail={low} weak={mid} pos-only={hi}]  error={er}", flush=True)
    if errs:
        print(f"  error segs: {errs}", flush=True)


def main():
    args = sys.argv[1:]
    retry = "--retry" in args
    seqs = processed_sequences()
    if args and not args[0].startswith("--"):
        seqs = [args[0]]
    if not seqs:
        print("output/ds/ 下没有 batch_log.csv", flush=True)
        return
    for seq in seqs:
        rows = dedupe_csv(seq)
        if not rows:
            continue
        if retry and any(r["status"] == "error" for r in rows):
            print(f"[{seq}] retrying error segs...", flush=True)
            r = subprocess.run(
                [PYTHON, os.path.join(SCRIPT_DIR, "process_input_batch.py"), seq],
                capture_output=True, text=True, timeout=4 * 3600)
            rows = dedupe_csv(seq)
        report(seq, rows)


if __name__ == "__main__":
    main()
