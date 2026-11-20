@echo off
rem 构建 BroadTrack Docker 镜像（首次构建会下载 CUDA 基础镜像 + LibTorch ~2GB，约需 20-60 分钟）
cd /d %~dp0
docker build . -t broadtrack
if %errorlevel% neq 0 (
    echo.
    echo [FAILED] 构建失败。常见原因：
    echo   1. Docker Desktop 未启动
    echo   2. WSL2 未安装好（运行 wsl --status 检查）
    exit /b 1
)
echo.
echo [OK] 镜像构建完成。测试: docker run --rm --gpus=all broadtrack broadtrack --help
