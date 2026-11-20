@echo off
rem 用法: run-broadtrack.cmd <图片文件夹> <输出json路径> [tripod文件路径]
rem 例:   run-broadtrack.cmd C:\data\frames C:\data\broadtrack.json
rem       run-broadtrack.cmd C:\data\frames C:\data\broadtrack.json C:\data\tripod.json
setlocal EnableDelayedExpansion

if "%~1"=="" (
    echo 用法: %~0 ^<图片文件夹^> ^<输出json路径^> [tripod文件路径]
    echo 图片文件夹内应为 %%06d.jpg 格式的连续帧，可用:
    echo   ffmpeg -i video.mp4 -q:v 1 frames\%%06d.jpg
    exit /b 1
)

set "FRAMES=%~f1"
set "OUTPUT=%~f2"
set "TRIPOD=%~f3"

rem 输出目录（json 会写到该目录下；容器内挂载为 /output）
set "OUTDIR=%~dp2"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
set "OUTNAME=%~nx2"

if "%TRIPOD%"=="" (
    docker run --rm --gpus=all -v "%FRAMES%:/frames" -v "%OUTDIR%:/output" broadtrack broadtrack --f /frames --o /output/%OUTNAME%
) else (
    docker run --rm --gpus=all -v "%FRAMES%:/frames" -v "%OUTDIR%:/output" -v "%TRIPOD%:/tripod.json:ro" broadtrack broadtrack --f /frames --o /output/%OUTNAME% --t /tripod.json
)
