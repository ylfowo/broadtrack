"""独立复现 C++ 投影：用 pass2 json 里的相机参数重投影球场线框，
叠加在输入帧上，并与模型检测线对照。
蓝 = json 参数重投影线框;  绿 = 模型检测线;  紫 = 两者重叠
用法: python reproject_check.py <帧路径> <json路径> <输出png>
"""
import sys
import json
import numpy as np
import torch
import cv2

MODEL = r"C:\Users\84261\Desktop\BroadTrack\models\tvcalib_model.pt"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def build_wireframe():
    """与 SoccerPitch3D 相同的球场线（105x68, 原点在中心）"""
    lines = []
    def add(pts): lines.append(np.array(pts, dtype=float))
    add([(-52.5, -34), (52.5, -34)])
    add([(-52.5, 34), (52.5, 34)])
    add([(-52.5, -34), (-52.5, 34)])
    add([(52.5, -34), (52.5, 34)])
    add([(0, -34), (0, 34)])
    th = np.linspace(0, 2*np.pi, 120)
    add([(9.15*np.cos(t), 9.15*np.sin(t)) for t in th])
    for sx in (-1, 1):
        x0 = sx * (52.5 - 16.5)
        add([(x0, -20.16), (x0, 20.16)])
        add([(x0, -20.16), (sx*52.5, -20.16)])
        add([(x0, 20.16), (sx*52.5, 20.16)])
    for sx in (-1, 1):
        x0 = sx * (52.5 - 5.5)
        add([(x0, -9.16), (x0, 9.16)])
        add([(x0, -9.16), (sx*52.5, -9.16)])
        add([(x0, 9.16), (sx*52.5, 9.16)])
    return lines

def camera_from_json(cp, W, H):
    """按 Camera.cpp 约定构造: cam = R^T (X - p), pixel = K * (x/z, y/z)"""
    p = np.array([cp['positionXMeters'], cp['positionYMeters'], cp['positionZMeters']])
    pan, tilt, roll = np.radians(cp['panDegrees']), np.radians(cp['tiltDegrees']), np.radians(cp['rollDegrees'])
    Rt = np.array([[1, 0, 0],
                   [0, np.cos(tilt), -np.sin(tilt)],
                   [0, np.sin(tilt), np.cos(tilt)]])
    Rp = np.array([[np.cos(pan), -np.sin(pan), 0],
                   [np.sin(pan), np.cos(pan), 0],
                   [0, 0, 1]])
    Rr = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), np.cos(roll), 0],
                   [0, 0, 1]])
    R = Rp @ Rt @ Rr
    f = (W/2) / np.tan(np.radians(cp['horizontalFieldOfViewDegrees'])/2)
    return R, p, f

def project_wireframe(R, p, f, W, H, lines):
    img = np.zeros((H, W), dtype=np.uint8)
    for line in lines:
        # 密集采样(0.1m)，与 C++ getWireframe(0.1) 一致
        dense = []
        for i in range(len(line) - 1):
            seg = line[i+1] - line[i]
            n = max(int(np.linalg.norm(seg) / 0.1), 1)
            for t in np.linspace(0, 1, n + 1)[:-1]:
                dense.append(line[i] + t * seg)
        dense.append(line[-1])
        prev = None
        for x, y in dense:
            X = np.array([x, y, 0.0])
            cam = R.T @ (X - p)
            if cam[2] <= 0.1:
                prev = None; continue
            u = f * cam[0]/cam[2] + W/2
            v = f * cam[1]/cam[2] + H/2
            if 0 <= u < W and 0 <= v < H:
                pt = (int(u), int(v))
                if prev is not None:
                    cv2.line(img, prev, pt, 255, 2)
                prev = pt
            else:
                prev = None
    return img

def run_detection(img):
    h, w = img.shape[:2]
    if h > w: new_w, new_h = 256, int(256.0*h/w)
    else: new_h, new_w = 256, int(256.0*w/h)
    r = cv2.resize(img, (new_w, new_h))
    model = torch.jit.load(MODEL, map_location='cpu')
    t = torch.from_numpy(r).permute(2,0,1).float()/255
    t = (t - torch.tensor(MEAN).view(3,1,1)) / torch.tensor(STD).view(3,1,1)
    with torch.no_grad():
        out = model(t.unsqueeze(0))
    res = out.toGenericDict()['out'] if hasattr(out,'toGenericDict') else out['out']
    return res.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)

def main(frame_path, json_path, out_path):
    frame_name = frame_path.replace('\\', '/').split('/')[-1]
    img = cv2.imread(frame_path)
    H, W = img.shape[:2]

    d = json.load(open(json_path))
    key = [k for k in d if k.endswith('/' + frame_name)]
    if not key:
        print(f'frame {frame_name} not found in json'); sys.exit(1)
    cp = d[key[0]]['cp']
    score = d[key[0]]['score']
    print(f'frame {frame_name}, score={score:.3f}, fov={cp["horizontalFieldOfViewDegrees"]:.1f}, '
          f'pan={cp["panDegrees"]:.1f}, tilt={cp["tiltDegrees"]:.1f}, pos=({cp["positionXMeters"]:.1f}, {cp["positionYMeters"]:.1f}, {cp["positionZMeters"]:.1f})')

    R, p, f = camera_from_json(cp, W, H)
    wf = project_wireframe(R, p, f, W, H, build_wireframe())

    seg = run_detection(img)
    seg_up = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
    det = (seg_up > 0).astype(np.uint8) * 255

    # 量化
    det_pts = (det > 0).sum()
    wf_pts = (wf > 0).sum()
    wf_dil = cv2.dilate(wf, np.ones((9,9), np.uint8))
    det_dil = cv2.dilate(det, np.ones((9,9), np.uint8))
    overlap = (wf_dil & det_dil).sum()
    print(f'wireframe pixels: {wf_pts}, detection pixels: {det_pts}, dilated overlap: {overlap}')

    # 可视化: 蓝=重投影线框, 绿=检测线, 紫=重叠
    vis = img.copy()
    vis[wf > 0] = (255, 0, 0)
    vis[det > 0] = (0, 255, 0)
    vis[(wf > 0) & (det > 0)] = (255, 0, 255)
    cv2.imwrite(out_path, vis)
    print(f'saved: {out_path}')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
