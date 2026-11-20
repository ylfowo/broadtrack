"""受控实验：合成标准广播视角球场图，验证 tvcalib 模型 + Python 推理管线。
用法: python synthetic_test.py <输出目录>
"""
import sys
import numpy as np
import torch
import cv2

MODEL = r"C:\Users\84261\Desktop\BroadTrack\models\tvcalib_model.pt"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def make_pitch_image(W=1024, H=576, fov=55., cam_pos=(0., 55., -12.), look_at=(0., 0., 0.)):
    """渲染合成球场：FIFA 标准线 + 看台高处机位针孔投影"""
    # 世界坐标：x=长轴 [-52.5,52.5], y=宽轴 [-34,34], z=0 地面
    lines = []  # 每条线是点列表
    def add_line(pts):
        lines.append(np.array(pts, dtype=float))

    # 边线+底线
    add_line([(-52.5, -34), (52.5, -34)])
    add_line([(-52.5, 34), (52.5, 34)])
    add_line([(-52.5, -34), (-52.5, 34)])
    add_line([(52.5, -34), (52.5, 34)])
    # 中线
    add_line([(0, -34), (0, 34)])
    # 中圈
    theta = np.linspace(0, 2*np.pi, 120)
    add_line([(9.15*np.cos(t), 9.15*np.sin(t)) for t in theta])
    # 禁区 (16.5m 深, 40.32m 宽)
    for sx in (-1, 1):
        x0 = sx * (52.5 - 16.5)
        add_line([(x0, -20.16), (x0, 20.16)])
        add_line([(x0, -20.16), (sx*52.5, -20.16)])
        add_line([(x0, 20.16), (sx*52.5, 20.16)])
    # 小禁区 (5.5m, 18.32m)
    for sx in (-1, 1):
        x0 = sx * (52.5 - 5.5)
        add_line([(x0, -9.16), (x0, 9.16)])
        add_line([(x0, -9.16), (sx*52.5, -9.16)])
        add_line([(x0, 9.16), (sx*52.5, 9.16)])
    # 点球点
    for sx in (-1, 1):
        add_line([(sx*(52.5-11), 0), (sx*(52.5-11), 0)])

    # 相机：cam1 先验 (0, 55, -12)，看向 (0,0,0)
    C = np.array(cam_pos, dtype=float)
    T = np.array(look_at, dtype=float)
    fwd = T - C; fwd = fwd / np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, -1.0])
    right = np.cross(fwd, world_up); right = right / np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.stack([right, up, fwd])  # 世界→相机

    # FOV 可变, 主点中心
    f = (W/2) / np.tan(np.radians(fov)/2)
    K = np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1]])

    def project(p):
        p = np.array([p[0], p[1], 0.0])
        pc = R @ (p - C)
        if pc[2] <= 0.1: return None
        uv = K @ pc
        return (uv[0]/uv[2], uv[1]/uv[2])

    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (70, 140, 55)  # 草地绿色 (BGR)
    for line in lines:
        pts = [project(p) for p in line]
        pts = [p for p in pts if p is not None and 0 <= p[0] < W and 0 <= p[1] < H]
        if len(pts) > 1:
            for i in range(len(pts)-1):
                cv2.line(img, (int(pts[i][0]), int(pts[i][1])),
                         (int(pts[i+1][0]), int(pts[i+1][1])), (255, 255, 255), 3)
    return img

def run_model(img):
    h, w = img.shape[:2]
    if h > w:
        new_w, new_h = 256, int(256.0 * h / w)
    else:
        new_h, new_w = 256, int(256.0 * w / h)
    resized = cv2.resize(img, (new_w, new_h))
    model = torch.jit.load(MODEL, map_location="cpu")
    t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    t = (t - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)
    with torch.no_grad():
        out = model(t.unsqueeze(0))
    result = out.toGenericDict()["out"] if hasattr(out, "toGenericDict") else out["out"]
    return result.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)

def main(outdir):
    import os
    os.makedirs(outdir, exist_ok=True)
    img = make_pitch_image()
    seg = run_model(img)
    H, W = img.shape[:2]
    seg_up = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)

    # 量化：模型线像素 vs 合成图中真实白线像素
    true_lines = np.all(img > 200, axis=2)
    pred_lines = seg_up > 0
    tp = (pred_lines & true_lines).sum()
    print(f"true line pixels: {true_lines.sum()}")
    print(f"model line pixels: {pred_lines.sum()}")
    print(f"overlap: {tp}  ({tp / max(true_lines.sum(), 1) * 100:.1f}% of true lines covered)")

    counts = np.bincount(seg.ravel(), minlength=29)
    print("detected classes:", [c for c in range(29) if counts[c] > 0])

    vis = img.copy()
    vis[pred_lines] = (0, 255, 0)
    vis[true_lines] = (255, 255, 255)
    cv2.imwrite(os.path.join(outdir, "synthetic_input.png"), img)
    cv2.imwrite(os.path.join(outdir, "synthetic_detection.png"), vis)
    print(f"saved to {outdir}")

if __name__ == "__main__":
    main(sys.argv[1])
