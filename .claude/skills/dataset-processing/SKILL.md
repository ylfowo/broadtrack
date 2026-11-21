---
name: dataset-processing
description: 足球比赛数据集批量标定标准流程（盘点→tripod估计→全量逆解→数据清洗→整合汇总）。当用户提供新比赛帧序列（input/ 下或任意路径）要求处理时使用；也可对已有输出做清洗/重跑。
---

# 数据集批量处理标准流程

BroadTrack 相机参数逆解的固定流水线，全部步骤已有脚本。

## 快速入口（首选）：一站式流水线

```
python scripts/run_pipeline.py                 # 全部 input/ 序列，一条命令跑完
python scripts/run_pipeline.py <seq>           # 单个序列
python scripts/run_pipeline.py --path <dir> [--name <n>]   # 任意位置序列
python scripts/run_pipeline.py --dry-run       # 只看计划不执行
python scripts/run_pipeline.py --no-tripod     # 跳过 tripod 估计
```

内部执行 5 步：盘点 → tripod 估计（已有跳过）→ 全量逆解（内部 3 并发）→ 清洗重试 → 汇总。
每步断点续跑：**中途任意时刻中断，重跑同一条命令即续跑**（已完成的段自动跳过）。
tripod 估计后自动校验（缺失则中止并报告）。
建议后台运行：一条命令挂后台，完成后人工抽查 montage 即可。

## 分步执行（调试/重跑单步用）

### 1. 盘点（只读）
```
python scripts/ds_common.py --list
```
输出每个 wide 序列的段数/帧数。自动识别两种嵌套结构：
`input/<比赛>/<比赛>_N_wide/frames` 和 `input/<比赛>/<比赛>_N_wide/<同名>/frames`。
确认：帧编号从 000001.jpg 连续（脚本会自动重排）、首帧分辨率（非 1080p 会自动上采样）。
新序列短名自动生成（目录词首 4 字母+机位号）；如需自定义名，在 `ds_common.py` 的 `KNOWN` 里登记。

### 2. 每序列 tripod 估计（~20 分钟/5 序列）
```
python scripts/estimate_input_tripods.py        # 全部（已有 tripod 自动跳过）
python scripts/estimate_input_tripods.py <seq>  # 单个
```
每序列采样首/中/末 3 段自由模式跑 → `compute_tripod.py` → `output/ds/<seq>_tripod.json`。
**收敛合理性校验（必做）**：
- X 接近 0（主看台中轴）或 ±36 附近
- Z 在 -9.5 ~ -26（看台高度不同；高位看台如安菲尔德 Z≈-26，塞尔赫斯特 Z≈-10）
- Y 在 30 ~ 100
- **同场两个机位应互相一致**（Y 差 <5、Z 差 <2）——一致性比绝对值更可信
不通过 → 换采样段重估（删 tripod json 重跑），或让用户确认机位。

### 3. 全量批量（逆解）
```
python scripts/process_input_batch.py <seq>     # 单序列，后台运行
python scripts/process_input_batch.py --all     # 全部序列
python scripts/process_input_batch.py --path <frames_dir> [--name <n>]
```
- **并发上限 3 个 worker**（RTX 4080，每进程 ~3GB 显存；超 3 会偶发 segfault rc=139）
- 断点续跑：batch_log.csv 跳过 ok/flagged，error 段重跑即可补
- 每段输出：`output/ds/<seq>/seg_XXX/{seg_XXX.json, vis/t_*.jpg, montage.png, stats.json}`
- 崩溃段（rc≠0）用清洗步骤重试一次，绝大多数是并发负载偶发

### 4. 数据清洗
```
python scripts/clean_batch.py <seq> --retry      # 去重+重试 error+分类报告
python scripts/clean_batch.py                    # 全部序列（不重试）
```
- 去重 batch_log.csv（同段保留最后一行）
- `--retry` 调 process_input_batch 补跑 error 段
- 标记段三分类：**fail**（score<0.25，检测失败/素材限制，不修复）、**weak**（0.25-0.45，素材偏难）、**pos-only**（≥0.45，仅 tripod 深度先验偏差，投影可用）

### 5. 整合汇总
```
python scripts/make_ds_summary.py
```
自动扫描 output/ds/*/batch_log.csv 生成 `output/ds/DS_INPUT_SUMMARY.md`：
总览表、tripod 收敛共识 vs 估计偏差表、标记段分析、错误段清单。
最后向用户报告 + 建议抽查 montage（优先看 flagged 段和低分序列）。

## 关键知识（不可违反）

- **1080p 硬编码**：评分画布 960×540 + homography×0.5 只在 1920×1080 一致。所有帧必须上采样（脚本已自动做 INTER_LANCZOS4），帧名必须 000001.jpg 起连续（脚本自动重排）。
- **tripod 软约束**：光轴穿球心 + 眼点距球心 ≤5m（CauchyLoss）；k1 已限 ±2.0（勿动——放开会畸变爆炸、线收束一点）。
- **score 与 vis 两条路径**：score 走 homography 不经畸变；vis 走 3D 投影+畸变。score 正常但 vis 错乱 → 查 k1。
- Docker 命令必须带 `--gpus=all`；挂载路径 Windows 侧用绝对路径（脚本已处理）。
- 输出布局与日志格式见各脚本 docstring；不要改变 batch_log.csv 字段名（清洗/汇总依赖）。

## 已知经验值

- 历史批次：watford/westham OK 均分 0.60；首批 4 场 0.486-0.563；cry_sou/manc_cry/westham_2 0.577-0.598（最佳）
- tripod 采样估计 vs 批量共识偏差：0.6-3.5m（正常），>5m 的标记段多为深度先验偏移
- 单段耗时 ~30-60s；8 序列 ~1000 段 3 并发约 4-5 小时

## 参考

- 项目记忆 `broadtrack-windows-setup`（完整踩坑史：k1 爆炸、眼点滑动、初始化等）
- 历史汇总 `output/ds/DS_INPUT_SUMMARY.md`、`output/DS_SUMMARY.md`
