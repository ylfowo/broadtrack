"""诊断：检查 tvcalib 线检测模型在输入帧上的表现。
用法: python debug_detection.py <帧路径> <输出png路径>
"""
import sys
import numpy as np
import torch
import cv2

MODEL = r"C:\Users\84261\Desktop\BroadTrack\models\tvcalib_model.pt"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def main(frame_path, out_path):
    img = cv2.imread(frame_path)  # BGR, 与 C++ cv::imread 一致
    h, w = img.shape[:2]
    # 与 LineSegmentationModel.cpp 相同的预处理
    if h > w:
        new_w, new_h = 256, int(256.0 * h / w)
    else:
        new_h, new_w = 256, int(256.0 * w / h)
    resized = cv2.resize(img, (new_w, new_h))

    model = torch.jit.load(MODEL, map_location="cpu")
    model.eval()
    t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    t = (t - mean) / std
    with torch.no_grad():
        out = model(t.unsqueeze(0))
    if hasattr(out, "toGenericDict"):
        result = out.toGenericDict()["out"]
    else:
        result = out["out"]
    seg = result.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)

    # 类别统计
    counts = np.bincount(seg.ravel(), minlength=29)
    print("class pixel counts (0=background, 1-28=lines):")
    for c in range(29):
        if counts[c] > 0:
            print(f"  class {c:2d}: {counts[c]}")

    # 放大分割图到原图尺寸用于可视化
    seg_up = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)

    # Canny 边缘（真实白线的位置参考）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)

    # 量化对比：模型线像素与 Canny 边缘的重叠
    line_mask = (seg_up > 0).astype(np.uint8)
    overlap = (line_mask & (edges > 0).astype(np.uint8)).sum()
    print(f"\nmodel line pixels: {line_mask.sum()} ({line_mask.mean()*100:.1f}% of frame)")
    print(f"canny edge pixels: {(edges > 0).sum()}")
    print(f"overlap: {overlap}")

    # 可视化：绿=模型检测线，红=Canny 边缘，黄=重叠
    vis = img.copy()
    vis[line_mask > 0] = (0, 255, 0)
    vis[edges > 0] = (0, 0, 255)
    vis[(line_mask > 0) & (edges > 0)] = (0, 255, 255)
    cv2.imwrite(out_path, vis)
    print(f"saved: {out_path}  (green=model lines, red=canny edges, yellow=overlap)")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
