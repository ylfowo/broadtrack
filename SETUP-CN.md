# BroadTrack 环境配置说明（Windows）

## 方案：Docker（Linux 容器）+ WSL2 + GPU 直通

本项目官方只提供 Linux 构建方式（依赖 Boost / OpenCV / Ceres / LibTorch-CUDA，且代码中硬编码了 Linux 路径）。
在 Windows 上通过 Docker Desktop（WSL2 后端）运行 Linux 容器，GPU 可直通（RTX 4080 + 驱动 591.74 支持 CUDA 12.4 容器），**无需安装 Linux 系统**。

## 环境要求清单

| 组件 | 状态 | 说明 |
|------|------|------|
| NVIDIA 驱动 | ✅ 591.74 | 支持 CUDA 13.1，向下兼容容器里的 CUDA 12.4 |
| GPU | ✅ RTX 4080 16GB | 显存足够跑两个模型 |
| 内存 | ✅ 64GB | 构建和推理都够 |
| Docker Desktop | ✅ 已装 28.5.2 | WSL2 后端，GPU 直通已验证 |
| WSL2 + Ubuntu-22.04 | ✅ 已装 | 2026-11-02 通过 MSI 手动安装（`wsl --update` 报 0x803fb015 的绕行方案） |
| 镜像加速器 | ✅ 已配置 | ~/.docker/daemon.json 配了 3 个国内镜像源（重启 Docker Desktop 生效） |
| broadtrack 镜像 | ✅ 已构建 | 已通过 GPU 冒烟测试（两模型成功加载到 GPU） |
| Python（tripod 脚本） | ✅ soccer 环境 | `C:\Users\84261\.conda\envs\soccer`（Python 3.10，依赖齐全） |
| 模型文件 | ✅ 已下载 | models/ 下两个 .pt（git LFS 拉取成功） |

## 安装步骤（✅ 已完成 2026-11-02）

### 1. 安装 WSL（需要管理员权限）

`wsl --update` 因系统 Windows 更新设置报错 0x803fb015，改用 MSI 手动安装：
- 下载：https://github.com/microsoft/WSL/releases/tag/2.9.13 → `wsl.2.9.13.0.x64.msi`
- 双击安装，然后（管理员 PowerShell）：
  ```
  wsl --install -d Ubuntu-22.04
  ```
- 装完重启电脑（如果提示的话），验证：`wsl --status` 不再报"未找到 WSL 2 内核文件"

### 2. 启动 Docker Desktop 并验证 GPU

启动 Docker Desktop，等右下角状态变成 Running，然后在 PowerShell 里：
```
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```
能看到 GPU 信息（RTX 4080）即 GPU 直通成功。

### 3. Docker Hub 网络问题（国内必读）

Docker Hub（registry.docker.io）在本机网络不通，两个绕行方案：
1. **镜像加速器**（已配好）：`~/.docker/daemon.json` 已写入 docker.1ms.run / docker.xuanyuan.me / docker.m.daocloud.io，重启 Docker Desktop 生效
2. **前缀拉取**：`docker pull docker.1ms.run/<镜像名>` 然后 `docker tag` 回原名

基础镜像已按方案 2 拉好并打了标签：
```
nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04   (13.1GB)
nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04  (5.47GB)
```

### 4. 构建镜像（✅ 已完成，约 25 分钟）

```
cd C:\Users\84261\Desktop\BroadTrack
build-docker.cmd
```
> 已对 Dockerfile 做了两处修改：
> 1. 基础镜像从 `nvcr.io` 换成 Docker Hub 的 `nvidia/cuda`（nvcr.io 需要 NGC 账号登录）
> 2. `make -j 32` 改为 `make -j$(nproc)`（按实际核心数编译）
>
> ✅ 镜像已构建并完成 GPU 冒烟测试：两个 torch 模型成功加载到 GPU（"CUDA is available! Moving model to GPU."），推理耗时毫秒级。

### 5. 运行

先把视频抽帧（代码要求 `%06d.jpg` 连续命名）：
```
ffmpeg -i video.mp4 -q:v 1 frames\%06d.jpg
```

第一次跑（无 tripod 文件，用默认相机参数，参照 README 的先验值）：

| 相机类型 | X   | Y   | Z   |
|----------|-----|-----|-----|
| cam1 |  0  |  55 | -12 |
| 16mr | 36  |  55 | -12 |
| 16ml | -36 |  55 | -12 |
| hbg  | -65 |  0  | -15 |

```
run-broadtrack.cmd C:\data\frames C:\data\broadtrack.json
```

### 6. 估计 tripod 参数（可选）

用第 4 步生成的标定文件跑 Python 脚本：
```
cd C:\Users\84261\Desktop\BroadTrack\scripts
C:\Users\84261\.conda\envs\soccer\python.exe compute_tripod.py -i C:\data\broadtrack.json -o C:\data\tripod.json
```

再带 tripod 文件重跑：
```
run-broadtrack.cmd C:\data\frames C:\data\broadtrack2.json C:\data\tripod.json
```

## 命令行参数速查

```
broadtrack --help
  -f  图片文件夹（%06d.jpg 连续帧）
  -o  输出 json（默认 ./broadtrack.json）
  -t  tripod 信息文件（由 compute_tripod.py 生成，含 sphere 字段）
  -l  TVCalib 模型路径（默认 /usr/share/broadtrack/tvcalib_model.pt）
  -k  关键点模型路径（默认 /usr/share/broadtrack/nbjw_keypoint_model.pt）
  -X -Y -Z  tripod 位置先验（默认 0, 90, -18）
  -b  人体 bbox 目录（可选）
  -r  输出可视化帧目录（可选）
```

## 常见问题

- **`docker: error during connect`** → Docker Desktop 没启动
- **`wsl --status` 提示"未找到 WSL 2 内核文件"** → MSI 没装好，重装后重启
- **容器里 `nvidia-smi` 报错** → Docker Desktop 设置里确认 WSL Integration 开启
- **构建时内存不足** → Docker Desktop 设置 → Resources 里调大 WSL 内存限制
