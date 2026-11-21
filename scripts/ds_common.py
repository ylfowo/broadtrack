"""BroadTrack 数据集批处理的公共工具：序列自动发现、短名生成、盘点。
支持两种结构：
  A. input/<比赛>/<比赛>_N_wide/frames/seg_XXX/*.jpg
  B. input/<比赛>/<比赛>_N_wide/<同名>/frames/seg_XXX/*.jpg（westham 双层嵌套）
"""
import os
import re
import sys

INPUT_ROOT = r"C:\Users\84261\Desktop\BroadTrack\input"

# 已知序列的显式短名（路径 -> 短名）；未登记的自动生成确定性短名
KNOWN = {
    r"input\2017-08-13_manchester-united-west-ham-united-premier-league-2017-2018\manchester-united-west-ham-united-premier-league-2017-2018_1_wide\frames": "manu_whu_1",
    r"input\2017-08-13_manchester-united-west-ham-united-premier-league-2017-2018\manchester-united-west-ham-united-premier-league-2017-2018_2_wide\frames": "manu_whu_2",
    r"input\2017-08-19_liverpool-fc-crystal-palace\liverpool-fc-crystal-palace_1_wide\frames": "liv_cry_1",
    r"input\2017-08-19_liverpool-fc-crystal-palace\liverpool-fc-crystal-palace_2_wide\frames": "liv_cry_2",
    r"input\2017-09-09_stoke-city-manchester-united-premier-league\stoke-city-manchester-united-premier-league_1_wide\frames": "stoke_manu_1",
    r"input\2017-09-09_stoke-city-manchester-united-premier-league\stoke-city-manchester-united-premier-league_2_wide\frames": "stoke_manu_2",
    r"input\2017-09-10_burnley-fc-crystal-palace\burnley-fc-crystal-palace_1_wide\frames": "burnley_cry_1",
    r"input\2017-09-10_burnley-fc-crystal-palace\burnley-fc-crystal-palace_2_wide\frames": "burnley_cry_2",
    r"input\2017-09-16_crystal-palace-southampton-fc\crystal-palace-southampton-fc_1_wide\frames": "cry_sou_1",
    r"input\2017-09-16_crystal-palace-southampton-fc\crystal-palace-southampton-fc_2_wide\frames": "cry_sou_2",
    r"input\2017-09-23_manchester-city-crystal-palace-premier-league-2017-2018\manchester-city-crystal-palace-premier-league-2017-2018_1_wide\frames": "manc_cry_1",
    r"input\2017-09-23_manchester-city-crystal-palace-premier-league-2017-2018\manchester-city-crystal-palace-premier-league-2017-2018_2_wide\frames": "manc_cry_2",
    r"input\2017-09-23_west-ham-united-tottenham-hotspur\west-ham-united-tottenham-hotspur_2_wide\west-ham-united-tottenham-hotspur_2_wide\frames": "westham_2",
}


def iter_sequence_dirs(input_root=INPUT_ROOT):
    """扫描 input/ 下所有 wide 序列，返回 [(frames_dir, match_dir, cam_dir)]。"""
    out = []
    if not os.path.isdir(input_root):
        return out
    for match_dir in sorted(os.listdir(input_root)):
        mdir = os.path.join(input_root, match_dir)
        if not os.path.isdir(mdir):
            continue
        for cam_dir in sorted(os.listdir(mdir)):
            cdir = os.path.join(mdir, cam_dir)
            if not os.path.isdir(cdir) or "_wide" not in cam_dir:
                continue
            for cand in (os.path.join(cdir, "frames"),
                         os.path.join(cdir, cam_dir, "frames")):
                if os.path.isdir(cand):
                    out.append((cand, match_dir, cam_dir))
                    break
    return out


def auto_short_name(match_dir, cam_dir):
    """由目录名生成确定性短名：去日期/联赛后缀，每词取前 4 字母 + 机位号。
    如 2017-09-23_manchester-city-..._1_wide -> manc_city_crys_pala_..._1（较长但唯一）。"""
    name = re.sub(r"^\d{4}-\d{2}-\d{2}_", "", match_dir)
    name = re.sub(r"-premier-league-2017-2018$", "", name)
    m = re.match(r".*_(\d+)_wide$", cam_dir)
    cam = m.group(1) if m else cam_dir
    words = [w[:4] for w in name.split("-") if w]
    return "_".join(words) + f"_{cam}"


def discover(input_root=INPUT_ROOT):
    """发现全部序列 -> {short_name: frames_dir}。显式登记名优先（按路径匹配）。"""
    known_by_path = {}
    for rel, name in KNOWN.items():
        known_by_path[os.path.normpath(os.path.join(
            os.path.dirname(input_root), rel))] = name
    seqs = {}
    for frames_dir, match_dir, cam_dir in iter_sequence_dirs(input_root):
        norm = os.path.normpath(frames_dir)
        name = known_by_path.get(norm) or auto_short_name(match_dir, cam_dir)
        seqs[name] = frames_dir
    return seqs


def count_segments(frames_dir):
    """返回 (段数, 总帧数)。"""
    if not os.path.isdir(frames_dir):
        return 0, 0
    n_seg = n_frames = 0
    for s in sorted(os.listdir(frames_dir)):
        sdir = os.path.join(frames_dir, s)
        if os.path.isdir(sdir):
            n_seg += 1
            n_frames += sum(1 for f in os.listdir(sdir)
                            if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return n_seg, n_frames


if __name__ == "__main__":
    if "--list" in sys.argv:
        seqs = discover()
        total_seg = total_fr = 0
        print(f"input/ 序列盘点: {len(seqs)} 个 wide 序列")
        for name, frames in seqs.items():
            ns, nf = count_segments(frames)
            total_seg += ns
            total_fr += nf
            print(f"  {name:20s} {ns:4d} 段 {nf:7d} 帧  {frames}")
        print(f"合计: {total_seg} 段 / {total_fr} 帧")
    else:
        print("用法: python ds_common.py --list   （盘点 input/ 下所有 wide 序列）")
