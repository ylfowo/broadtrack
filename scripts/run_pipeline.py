"""一站式流水线：盘点 → tripod 估计 → 全量逆解（内部 3 并发）→ 清洗重试 → 汇总。
每步自带断点续跑，任意时刻中断后重跑同一条命令即可接着跑。

用法:
  python run_pipeline.py                     全部 input/ 序列
  python run_pipeline.py <seq>               单个序列
  python run_pipeline.py --path <frames_dir> [--name <n>]
  python run_pipeline.py --dry-run           只看计划，不执行
  python run_pipeline.py --no-tripod         跳过 tripod 估计（用已有 tripod）
"""
import os
import sys
import json
import subprocess
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

from ds_common import discover

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DS = r"C:\Users\84261\Desktop\BroadTrack\output\ds"
WORKERS = 3  # RTX 4080 并发上限，勿调大（超 3 偶发 segfault）


def py(script, *args, capture=False, timeout=None):
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, script), *args]
    print(f"\n===== {script} {' '.join(args)} =====", flush=True)
    if capture:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return subprocess.run(cmd, timeout=timeout)


def tripod_ok(name):
    return os.path.isfile(os.path.join(OUT_DS, f"{name}_tripod.json"))


def seg_done_count(name):
    p = os.path.join(OUT_DS, name, "batch_log.csv")
    if not os.path.isfile(p):
        return 0, 0
    with open(p, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] in ("ok", "flagged")]
    return len(rows), sum(1 for r in rows if r["status"] == "flagged")


def plan(seqs, need_tripod):
    print("流水线计划:")
    for name in seqs:
        t = "估tripod" if (need_tripod and not tripod_ok(name)) else "用已有tripod"
        done, fl = seg_done_count(name)
        print(f"  {name:24s} {t:12s} 已完成 {done:4d} 段 (flagged {fl})")
    print(f"并发 worker: {WORKERS}", flush=True)


def run_batch(names):
    if not names:
        return
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(py, "process_input_batch.py", n): n for n in names}
        for fut in as_completed(futs):
            n = futs[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[{n}] worker exception: {e}", flush=True)


def docker_ready():
    """预检 docker：未安装/不在 PATH/守护进程未启动时给出明确提示。"""
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            return True
        msg = r.stderr.strip().splitlines()
        if "daemon" in " ".join(msg).lower() or "pipe" in " ".join(msg).lower():
            print("!! Docker Desktop 未启动：请先打开 Docker Desktop（等鲸鱼图标变绿）再运行", flush=True)
        else:
            print(f"!! docker 不可用: {' '.join(msg[:3])}", flush=True)
        return False
    except FileNotFoundError:
        print("!! 找不到 docker 命令：确认 Docker Desktop 已安装且 PATH 含其 bin 目录", flush=True)
        return False


def main():
    args = sys.argv[1:]
    if "--dry-run" not in args and not docker_ready():
        return
    seqs = discover()
    if "--path" in args:
        i = args.index("--path")
        frames_dir = args[i + 1]
        name = None
        if "--name" in args:
            name = args[args.index("--name") + 1]
        if name is None:
            name = os.path.basename(os.path.normpath(frames_dir))
        seqs = {name: frames_dir}
        names = [name]
    elif args and not args[0].startswith("--"):
        names = [args[0]]
    else:
        names = sorted(seqs)
    names = [n for n in names if n in seqs]
    if not names:
        print("没有可处理的序列（ds_common.py --list 查看）", flush=True)
        return

    if "--dry-run" in args:
        plan(names, need_tripod=True)
        return

    # 1. 盘点
    r = py("ds_common.py", "--list", capture=True)
    if r is not None and r.stdout:
        print(r.stdout, flush=True)
    # 2. tripod 估计（已有自动跳过）
    if "--no-tripod" not in args:
        py("estimate_input_tripods.py", *names)
        missing = [n for n in names if not tripod_ok(n)]
        if missing:
            print(f"!! tripod 仍缺失，中止: {missing}", flush=True)
            return
    # 3. 全量逆解（内部并发，断点续跑）
    run_batch(names)
    # 4. 清洗 + 重试错误段
    py("clean_batch.py", *names, "--retry")
    # 5. 汇总
    py("make_ds_summary.py")
    print("\n流水线完成。汇总: output/ds/DS_INPUT_SUMMARY.md", flush=True)


if __name__ == "__main__":
    main()
