"""为 input/ 下 8 个 wide 序列估计 tripod：
每序列取 3 个分布均匀的段 -> 上采样 -> 自由模式跑 -> 合并 -> compute_tripod.py
输出: output/ds/<seq>_tripod.json
"""
import os
import sys
import json
import subprocess
import shutil

import cv2

SEQUENCES = {
    "manu_whu_1": r"C:\Users\84261\Desktop\BroadTrack\input\2017-08-13_manchester-united-west-ham-united-premier-league-2017-2018\manchester-united-west-ham-united-premier-league-2017-2018_1_wide\frames",
    "manu_whu_2": r"C:\Users\84261\Desktop\BroadTrack\input\2017-08-13_manchester-united-west-ham-united-premier-league-2017-2018\manchester-united-west-ham-united-premier-league-2017-2018_2_wide\frames",
    "liv_cry_1": r"C:\Users\84261\Desktop\BroadTrack\input\2017-08-19_liverpool-fc-crystal-palace\liverpool-fc-crystal-palace_1_wide\frames",
    "liv_cry_2": r"C:\Users\84261\Desktop\BroadTrack\input\2017-08-19_liverpool-fc-crystal-palace\liverpool-fc-crystal-palace_2_wide\frames",
    "stoke_manu_1": r"C:\Users\84261\Desktop\BroadTrack\input\2017-09-09_stoke-city-manchester-united-premier-league\stoke-city-manchester-united-premier-league_1_wide\frames",
    "stoke_manu_2": r"C:\Users\84261\Desktop\BroadTrack\input\2017-09-09_stoke-city-manchester-united-premier-league\stoke-city-manchester-united-premier-league_2_wide\frames",
    "burnley_cry_1": r"C:\Users\84261\Desktop\BroadTrack\input\2017-09-10_burnley-fc-crystal-palace\burnley-fc-crystal-palace_1_wide\frames",
    "burnley_cry_2": r"C:\Users\84261\Desktop\BroadTrack\input\2017-09-10_burnley-fc-crystal-palace\burnley-fc-crystal-palace_2_wide\frames",
    "cry_sou_1": r"C:\Users\84261\Desktop\BroadTrack\input\2017-09-16_crystal-palace-southampton-fc\crystal-palace-southampton-fc_1_wide\frames",
    "cry_sou_2": r"C:\Users\84261\Desktop\BroadTrack\input\2017-09-16_crystal-palace-southampton-fc\crystal-palace-southampton-fc_2_wide\frames",
    "manc_cry_1": r"C:\Users\84261\Desktop\BroadTrack\input\2017-09-23_manchester-city-crystal-palace-premier-league-2017-2018\manchester-city-crystal-palace-premier-league-2017-2018_1_wide\frames",
    "manc_cry_2": r"C:\Users\84261\Desktop\BroadTrack\input\2017-09-23_manchester-city-crystal-palace-premier-league-2017-2018\manchester-city-crystal-palace-premier-league-2017-2018_2_wide\frames",
    "westham_2": r"C:\Users\84261\Desktop\BroadTrack\input\2017-09-23_west-ham-united-tottenham-hotspur\west-ham-united-tottenham-hotspur_2_wide\west-ham-united-tottenham-hotspur_2_wide\frames",
}

PREP_ROOT = r"C:\Users\84261\Desktop\BroadTrack\input_1080\tripod_est"
OUT_ROOT = r"C:\Users\84261\Desktop\BroadTrack\output"
TRIPOD_OUT = r"C:\Users\84261\Desktop\BroadTrack\output\ds"
PYTHON = r"C:\Users\84261\.conda\envs\soccer\python.exe"
COMPUTE_TRIPOD = r"C:\Users\84261\Desktop\BroadTrack\scripts\compute_tripod.py"


def prep_seg(name, seg, srcdir):
    files = sorted(f for f in os.listdir(srcdir) if f.lower().endswith(".jpg"))
    if not files:
        return None
    prepdir = os.path.join(PREP_ROOT, name, seg)
    if os.path.isdir(prepdir):
        shutil.rmtree(prepdir)
    os.makedirs(prepdir)
    for i, f in enumerate(files, 1):
        im = cv2.imread(os.path.join(srcdir, f))
        if im is None:
            continue
        h, w = im.shape[:2]
        if (w, h) != (1920, 1080):
            im = cv2.resize(im, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(prepdir, f"{i:06d}.jpg"), im)
    return len(files)


def main():
    os.makedirs(TRIPOD_OUT, exist_ok=True)
    for name, frames_dir in SEQUENCES.items():
        if os.path.isfile(os.path.join(TRIPOD_OUT, f"{name}_tripod.json")):
            print(f"[{name}] tripod exists, skip", flush=True)
            continue
        segs = sorted(s for s in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, s)))
        if not segs:
            print(f"[{name}] no segs!", flush=True)
            continue
        # 采样: 首/中/末
        sample_idx = [0, len(segs) // 2, len(segs) - 1]
        samples = [segs[i] for i in sample_idx]
        merged = {}
        for seg in samples:
            srcdir = os.path.join(frames_dir, seg)
            n = prep_seg(name, seg, srcdir)
            if not n:
                print(f"[{name}] {seg}: empty, skip", flush=True)
                continue
            out_json = os.path.join(OUT_ROOT, "tripod_est", name, f"{seg}.json")
            os.makedirs(os.path.dirname(out_json), exist_ok=True)
            cmd = ["docker", "run", "--rm", "--gpus=all",
                   "-v", f"{os.path.join(PREP_ROOT, name)}:/input",
                   "-v", f"{OUT_ROOT}:/output",
                   "broadtrack", "broadtrack",
                   "--f", f"/input/{seg}",
                   "--o", f"/output/tripod_est/{name}/{seg}.json"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if r.returncode != 0 or not os.path.isfile(out_json):
                print(f"[{name}] {seg}: FAILED rc={r.returncode}", flush=True)
                continue
            with open(out_json) as f:
                merged.update(json.load(f))
            print(f"[{name}] {seg}: done ({n} frames)", flush=True)

        if len(merged) < 20:
            print(f"[{name}] too few frames ({len(merged)}), skip tripod", flush=True)
            continue
        merged_path = os.path.join(OUT_ROOT, "tripod_est", name, "merged_pass1.json")
        with open(merged_path, "w") as f:
            json.dump(merged, f, indent=2)
        tripod_path = os.path.join(TRIPOD_OUT, f"{name}_tripod.json")
        r = subprocess.run(
            [PYTHON, COMPUTE_TRIPOD, "-i", merged_path, "-o", tripod_path,
             "--iou", "0.45", "--tilt-window", "3.0", "--cos-threshold", "0.95"],
            capture_output=True, text=True, timeout=600)
        if os.path.isfile(tripod_path):
            with open(tripod_path) as f:
                t = json.load(f)
            print(f"[{name}] TRIPOD center={t['sphere']['center']} (from {len(merged)} frames)", flush=True)
        else:
            print(f"[{name}] tripod estimation failed: {r.stdout[-300:]} {r.stderr[-300:]}", flush=True)

    print("ALL TRIPODS DONE", flush=True)


if __name__ == "__main__":
    main()
