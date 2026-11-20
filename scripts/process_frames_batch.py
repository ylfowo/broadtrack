"""批量处理 input/frames 下所有段（按 group_0 tripod）。
每段输出到 output/frames_batch/seg_XXX/:
  - seg_XXX_1080.json  (相机参数)
  - vis/t_*.jpg        (逐帧投影图)
  - montage.png        (采样拼图)
  - stats.json         (统计: 分数/位置/与 group_0 偏差)
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

SRC = r"C:\Users\84261\Desktop\BroadTrack\input\frames"
PREP = r"C:\Users\84261\Desktop\BroadTrack\input_1080\frames_batch"
OUT_ROOT = r"C:\Users\84261\Desktop\BroadTrack\output"
OUT = os.path.join(OUT_ROOT, "frames_batch")
TRIPOD_CONTAINER_DEFAULT = "/output/tripod_group_0.json"
GROUP0 = np.array([0.06, 50.57, -13.28])

MAPPING_PATH = os.path.join(OUT_ROOT, "seg_tripod_mapping.json")


def tripod_for_seg(seg):
    """从映射表取该段的 tripod 文件(容器内路径)，默认 group_0。"""
    try:
        with open(MAPPING_PATH) as f:
            m = json.load(f)
        host_path = m.get(seg)
        if host_path:
            return "/output/" + os.path.basename(host_path)
    except Exception:
        pass
    return TRIPOD_CONTAINER_DEFAULT

LOG = os.path.join(OUT, "batch_log.csv")


def prep_seg(seg):
    """检查分辨率并重编号，输出到 PREP/seg/000001.jpg..."""
    srcdir = os.path.join(SRC, seg)
    files = sorted(f for f in os.listdir(srcdir) if f.lower().endswith(".jpg"))
    if not files:
        return None, "no jpg files"
    prepdir = os.path.join(PREP, seg)
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


def run_broadtrack(seg):
    outdir = os.path.join(OUT, seg)
    os.makedirs(os.path.join(outdir, "vis", "t_"), exist_ok=True)
    cmd = [
        "docker", "run", "--rm", "--gpus=all",
        "-v", f"{PREP}:/input",
        "-v", f"{OUT_ROOT}:/output",
        "broadtrack", "broadtrack",
        "--f", f"/input/{seg}",
        "--o", f"/output/frames_batch/{seg}/{seg}_1080.json",
        "--t", tripod_for_seg(seg),
        "--r", f"/output/frames_batch/{seg}/vis/",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    return r.returncode, r.stdout[-2000:] if r.stdout else ""


def make_stats_and_montage(seg):
    from PIL import Image, ImageDraw
    jp = os.path.join(OUT, seg, f"{seg}_1080.json")
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
    dev = float(np.linalg.norm(pmean - GROUP0))
    stats = {
        "seg": seg,
        "n_frames": len(keys),
        "score_mean": round(float(np.mean(scores)), 3),
        "score_min": round(float(min(scores)), 3),
        "pos_mean": [round(float(x), 2) for x in pmean],
        "pos_std": [round(float(x), 2) for x in pos.std(axis=0)],
        "group0_deviation_m": round(dev, 2),
        "flagged": bool(dev > 5.0 or np.mean(scores) < 0.45),
    }
    with open(os.path.join(OUT, seg, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 拼图: 最多 8 帧采样
    visdir = os.path.join(OUT, seg, "vis", "t_")
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
            grid.save(os.path.join(OUT, seg, "montage.png"))
    except Exception as e:
        print(f"  [warn] montage failed: {e}")
    return stats


def main():
    os.makedirs(OUT, exist_ok=True)
    segs = sorted(os.listdir(SRC))
    segs = [s for s in segs if os.path.isdir(os.path.join(SRC, s))]
    print(f"total segs: {len(segs)}", flush=True)

    fieldnames = ["seg", "status", "n_frames", "score_mean", "score_min",
                  "pos_x", "pos_y", "pos_z", "dev_m", "flagged", "seconds"]
    log_existed = os.path.isfile(LOG)
    with open(LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not log_existed:
            writer.writeheader()
        done = set()
        with open(LOG, newline="") as rf:
            for row in csv.DictReader(rf):
                if row["status"] in ("ok", "flagged"):
                    done.add(row["seg"])

        for si, seg in enumerate(segs, 1):
            if seg in done:
                print(f"[{si}/{len(segs)}] {seg}: skip (already done)", flush=True)
                continue
            t0 = time.time()
            row = {"seg": seg, "status": "error", "n_frames": 0,
                   "score_mean": "", "score_min": "", "pos_x": "", "pos_y": "",
                   "pos_z": "", "dev_m": "", "flagged": "", "seconds": 0}
            try:
                _, prep_info = prep_seg(seg)
                print(f"[{si}/{len(segs)}] {seg}: prep ({prep_info})", flush=True)
                rc, _ = run_broadtrack(seg)
                stats = make_stats_and_montage(seg)
                if rc != 0 or stats is None:
                    row["status"] = "error"
                    print(f"[{si}/{len(segs)}] {seg}: FAILED rc={rc}", flush=True)
                else:
                    row.update({
                        "status": "flagged" if stats["flagged"] else "ok",
                        "n_frames": stats["n_frames"],
                        "score_mean": stats["score_mean"],
                        "score_min": stats["score_min"],
                        "pos_x": stats["pos_mean"][0],
                        "pos_y": stats["pos_mean"][1],
                        "pos_z": stats["pos_mean"][2],
                        "dev_m": stats["group0_deviation_m"],
                        "flagged": stats["flagged"],
                    })
                    tag = " <-- FLAGGED" if stats["flagged"] else ""
                    print(f"[{si}/{len(segs)}] {seg}: score={stats['score_mean']} "
                          f"dev={stats['group0_deviation_m']}m{tag}", flush=True)
            except Exception as e:
                print(f"[{si}/{len(segs)}] {seg}: EXCEPTION {e}", flush=True)
                traceback.print_exc()
            row["seconds"] = int(time.time() - t0)
            writer.writerow(row)
            f.flush()

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
