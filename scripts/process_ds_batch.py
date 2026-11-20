"""批量处理 D:\\capture 数据集 (watford / westham)。
每段: 上采样1080p -> broadtrack(数据集tripod约束+彩色投影) -> stats+montage
输出: output/ds_<name>/seg_XXX/{seg_XXX.json, vis/t_*.jpg, montage.png, stats.json}
用法: python process_ds_batch.py <name>
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

DATASETS = {
    "watford": {
        "src": r"D:\capture\dataset\2017-08-12_watford-fc-liverpool-fc\watford-fc-liverpool-fc_2_wide\frames2",
        "tripod": r"C:\Users\84261\Desktop\BroadTrack\output\ds_test\tripod_watford.json",
    },
    "westham": {
        "src": r"D:\capture\dataset\2017-09-23_west-ham-united-tottenham-hotspur\west-ham-united-tottenham-hotspur_1_wide\frames3",
        "tripod": r"C:\Users\84261\Desktop\BroadTrack\output\ds_test\tripod_westham.json",
    },
}

OUT_ROOT = r"C:\Users\84261\Desktop\BroadTrack\output"
PREP_ROOT = r"C:\Users\84261\Desktop\BroadTrack\input_1080"
GROUP_CENTER = {"watford": np.array([-0.08, 48.50, -12.72]),
                "westham": np.array([-0.08, 96.52, -15.73])}


def prep_seg(name, seg):
    srcdir = os.path.join(DATASETS[name]["src"], seg)
    files = sorted(f for f in os.listdir(srcdir) if f.lower().endswith(".jpg"))
    if not files:
        return None, "no jpg files"
    prepdir = os.path.join(PREP_ROOT, f"ds_{name}", seg)
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
    outdir = os.path.join(OUT_ROOT, f"ds_{name}", seg)
    os.makedirs(os.path.join(outdir, "vis", "t_"), exist_ok=True)
    tripod_host = DATASETS[name]["tripod"]
    cmd = [
        "docker", "run", "--rm", "--gpus=all",
        "-v", f"{os.path.join(PREP_ROOT, 'ds_' + name)}:/input",
        "-v", f"{OUT_ROOT}:/output",
        "-v", f"{tripod_host}:/tripod.json:ro",
        "broadtrack", "broadtrack",
        "--f", f"/input/{seg}",
        "--o", f"/output/ds_{name}/{seg}/{seg}.json",
        "--t", "/tripod.json",
        "--r", f"/output/ds_{name}/{seg}/vis/",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    return r.returncode


def make_stats_and_montage(name, seg):
    from PIL import Image, ImageDraw
    jp = os.path.join(OUT_ROOT, f"ds_{name}", seg, f"{seg}.json")
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
    dev = float(np.linalg.norm(pmean - GROUP_CENTER[name]))
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
    with open(os.path.join(OUT_ROOT, f"ds_{name}", seg, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    visdir = os.path.join(OUT_ROOT, f"ds_{name}", seg, "vis", "t_")
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
            grid.save(os.path.join(OUT_ROOT, f"ds_{name}", seg, "montage.png"))
    except Exception as e:
        print(f"  [warn] montage failed: {e}")
    return stats


def main():
    name = sys.argv[1]
    outbase = os.path.join(OUT_ROOT, f"ds_{name}")
    os.makedirs(outbase, exist_ok=True)
    segs = sorted(os.listdir(DATASETS[name]["src"]))
    segs = [s for s in segs if os.path.isdir(os.path.join(DATASETS[name]["src"], s))]
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
                _, prep_info = prep_seg(name, seg)
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


if __name__ == "__main__":
    main()
