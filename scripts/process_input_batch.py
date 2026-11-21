"""批量处理 input/ 下 wide 序列（嵌套路径结构，自动发现）。
每段: 上采样1080p -> broadtrack(该序列tripod约束+彩色投影) -> stats+montage
输出: output/ds/<seq>/seg_XXX/{seg_XXX.json, vis/t_*.jpg, montage.png, stats.json}
用法:
  python process_input_batch.py <seq短名>   单个序列（断点续跑，跳过 ok/flagged）
  python process_input_batch.py --all       处理 input/ 下全部序列
  python process_input_batch.py --path <frames_dir> [--name <n>]   任意位置序列
"""
import os
import sys
import json
import csv
import time
import subprocess
import shutil
import traceback

import cv2
import numpy as np

from ds_common import discover

OUT_ROOT = r"C:\Users\84261\Desktop\BroadTrack\output"
PREP_ROOT = r"C:\Users\84261\Desktop\BroadTrack\input_1080\ds_input"
TRIPOD_DIR = os.path.join(OUT_ROOT, "ds")


def tripod_center(name):
    p = os.path.join(TRIPOD_DIR, f"{name}_tripod.json")
    if os.path.isfile(p):
        with open(p) as f:
            return np.array(json.load(f)["sphere"]["center"])
    return None


def prep_seg(name, seg, srcdir):
    files = sorted(f for f in os.listdir(srcdir) if f.lower().endswith(".jpg"))
    if not files:
        return None, "no jpg files"
    prepdir = os.path.join(PREP_ROOT, name, seg)
    if os.path.isdir(prepdir):
        shutil.rmtree(prepdir)
    os.makedirs(prepdir)
    n = 0
    for i, f in enumerate(files, 1):
        im = cv2.imread(os.path.join(srcdir, f))
        if im is None:
            continue
        h, w = im.shape[:2]
        if (w, h) != (1920, 1080):
            im = cv2.resize(im, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(prepdir, f"{i:06d}.jpg"), im)
        n += 1
    return prepdir, f"{n} frames"


def run_broadtrack(name, seg):
    outdir = os.path.join(OUT_ROOT, "ds", name, seg)
    os.makedirs(os.path.join(outdir, "vis", "t_"), exist_ok=True)
    tripod_host = os.path.join(TRIPOD_DIR, f"{name}_tripod.json")
    cmd = [
        "docker", "run", "--rm", "--gpus=all",
        "-v", f"{os.path.join(PREP_ROOT, name)}:/input",
        "-v", f"{OUT_ROOT}:/output",
        "-v", f"{tripod_host}:/tripod.json:ro",
        "broadtrack", "broadtrack",
        "--f", f"/input/{seg}",
        "--o", f"/output/ds/{name}/{seg}/{seg}.json",
        "--t", "/tripod.json",
        "--r", f"/output/ds/{name}/{seg}/vis/",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    return r.returncode


def make_stats_and_montage(name, seg):
    from PIL import Image, ImageDraw
    center = tripod_center(name)
    jp = os.path.join(OUT_ROOT, "ds", name, seg, f"{seg}.json")
    if not os.path.isfile(jp):
        return None
    with open(jp) as f:
        d = json.load(f)
    keys = sorted(d.keys())
    if not keys:
        return None
    scores = [d[k]["score"] for k in keys]
    pos = np.array([[d[k]["cp"]["positionXMeters"],
                     d[k]["cp"]["positionYMeters"],
                     d[k]["cp"]["positionZMeters"]] for k in keys])
    pmean = pos.mean(axis=0)
    dev = float(np.linalg.norm(pmean - center)) if center is not None else -1.0
    stats = {
        "seg": seg,
        "n_frames": len(keys),
        "score_mean": round(float(np.mean(scores)), 3),
        "score_min": round(float(min(scores)), 3),
        "pos_mean": [round(float(x), 2) for x in pmean],
        "pos_std": [round(float(x), 2) for x in pos.std(axis=0)],
        "tripod_deviation_m": round(dev, 2),
        "flagged": bool(dev > 5.0 or np.mean(scores) < 0.45),
    }
    with open(os.path.join(OUT_ROOT, "ds", name, seg, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    visdir = os.path.join(OUT_ROOT, "ds", name, seg, "vis", "t_")
    try:
        idx = list(range(0, len(keys), max(len(keys) // 8, 1)))[:8]
        imgs = []
        for i in idx:
            fn = keys[i].split("/")[-1]
            p = os.path.join(visdir, fn)
            if not os.path.isfile(p):
                continue
            im = Image.open(p)
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, 140, 20], fill="black")
            dr.text((4, 2), f"{fn} IoU={d[keys[i]]['score']:.3f}",
                    fill="red" if d[keys[i]]["score"] < 0.5 else "green")
            imgs.append(im)
        if imgs:
            w, h = imgs[0].size
            cols = 4
            rows = (len(imgs) + cols - 1) // cols
            grid = Image.new("RGB", (w * cols, h * rows), "white")
            for n, im in enumerate(imgs):
                grid.paste(im, ((n % cols) * w, (n // cols) * h))
            grid.save(os.path.join(OUT_ROOT, "ds", name, seg, "montage.png"))
    except Exception as e:
        print(f"  [warn] montage failed: {e}")
    return stats


def process_one(name, frames_dir):
    outbase = os.path.join(OUT_ROOT, "ds", name)
    os.makedirs(outbase, exist_ok=True)
    if tripod_center(name) is None:
        print(f"[{name}] tripod file missing: {TRIPOD_DIR}\\{name}_tripod.json", flush=True)
        return
    segs = sorted(s for s in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, s)))
    print(f"[{name}] total segs: {len(segs)}", flush=True)

    log_path = os.path.join(outbase, "batch_log.csv")
    fieldnames = ["seg", "status", "n_frames", "score_mean", "score_min",
                  "pos_x", "pos_y", "pos_z", "dev_m", "flagged", "seconds"]
    log_existed = os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not log_existed:
            writer.writeheader()
        done = set()
        with open(log_path, newline="") as rf:
            for row in csv.DictReader(rf):
                if row["status"] in ("ok", "flagged"):
                    done.add(row["seg"])

        for si, seg in enumerate(segs, 1):
            if seg in done:
                print(f"[{name}][{si}/{len(segs)}] {seg}: skip", flush=True)
                continue
            t0 = time.time()
            row = {"seg": seg, "status": "error", "n_frames": 0,
                   "score_mean": "", "score_min": "", "pos_x": "", "pos_y": "",
                   "pos_z": "", "dev_m": "", "flagged": "", "seconds": 0}
            try:
                srcdir = os.path.join(frames_dir, seg)
                _, prep_info = prep_seg(name, seg, srcdir)
                print(f"[{name}][{si}/{len(segs)}] {seg}: prep ({prep_info})", flush=True)
                rc = run_broadtrack(name, seg)
                stats = make_stats_and_montage(name, seg)
                if rc != 0 or stats is None:
                    row["status"] = "error"
                    print(f"[{name}][{si}/{len(segs)}] {seg}: FAILED rc={rc}", flush=True)
                else:
                    row.update({
                        "status": "flagged" if stats["flagged"] else "ok",
                        "n_frames": stats["n_frames"],
                        "score_mean": stats["score_mean"],
                        "score_min": stats["score_min"],
                        "pos_x": stats["pos_mean"][0],
                        "pos_y": stats["pos_mean"][1],
                        "pos_z": stats["pos_mean"][2],
                        "dev_m": stats["tripod_deviation_m"],
                        "flagged": stats["flagged"],
                    })
                    tag = " <-- FLAGGED" if stats["flagged"] else ""
                    print(f"[{name}][{si}/{len(segs)}] {seg}: score={stats['score_mean']} "
                          f"dev={stats['tripod_deviation_m']}m{tag}", flush=True)
            except Exception as e:
                print(f"[{name}][{si}/{len(segs)}] {seg}: EXCEPTION {e}", flush=True)
                traceback.print_exc()
            row["seconds"] = int(time.time() - t0)
            writer.writerow(row)
            f.flush()

    print(f"[{name}] ALL DONE", flush=True)


def main():
    args = sys.argv[1:]
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
    elif "--all" in args:
        names = sorted(seqs)
    elif args:
        names = [args[0]]
    else:
        print(__doc__)
        return
    for name in names:
        if name not in seqs:
            print(f"[{name}] unknown sequence (run ds_common.py --list)", flush=True)
            continue
        process_one(name, seqs[name])


if __name__ == "__main__":
    main()
