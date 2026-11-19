# bench：难场景基准测试（Python / PyTorch）

用来在加噪声、少样本、跨工况场景下比较识别网络。结果和结论见 `../results/bench_20261119/README.md`。

## 文件

| 文件 | 作用 |
| --- | --- |
| `tfr.py` | 时频图生成（同步提取变换 + 面积平均缩放 + jet 颜色），与 MATLAB 版逐像素相同 |
| `make_banks.py` | 从东南大学原始 CSV 生成各工况、各噪声水平的时频图库（CPU 密集，已算好） |
| `models.py` | `cnn`（v2 基线）、`cnn_wide`（对照）、`resnet`、`eca_resnet` |
| `lssvm.py` | BKA 优化 LS-SVM，逐行对应 MATLAB 版，预测结果与 Octave 版完全一致 |
| `run_bench.py` | 跑基准测试，结果逐行写入 CSV，可中断续跑 |
| `summarize.py` | 汇总均值 ± 标准差和配对 Wilcoxon 检验，只依赖 numpy |

## 环境

Python 3.9+、numpy、PyTorch（CUDA 版）。`make_banks.py` 导出 .mat 时需要 scipy，其余不需要。

## 在本机复现（RTX 2060）

1. 从数据分支下载图库，把 `banks/` 文件夹放到本目录下（`bench/banks/20_0_clean.npz` 等）。
2. 在 PowerShell 里进入 `bench` 目录：

```powershell
# 先跑一组试试（约 1 分钟）
python run_bench.py --device cuda --scenarios snr5 --models cnn,resnet --seeds 1

# 完整复现（6 个场景 x 3 个模型 x 10 个种子 = 180 组）
python run_bench.py --device cuda

# 加宽 CNN 对照
python run_bench.py --device cuda --scenarios snr10,snr5,snr0 --models cnn_wide

# 汇总
python summarize.py results/bench.csv --out results/bench_summary.md
```

- 默认只用 4 个 CPU 线程（`--threads 4`），不会让笔记本卡住。BKA-LSSVM 阶段在 CPU 上算，每组约 20–40 秒。
- 中途关掉没关系，再运行同一条命令会跳过已完成的组。
- 只想比较网络、不跑 LSSVM，加 `--no_lssvm`，速度快很多。
- GPU 上 cuDNN 有少量不确定性，数值和 `results/bench_20261119/bench.csv` 不会完全一样，但均值应在 1 个标准差以内。

## 场景名

`<图库>[_few<n>][+mix][+adabn][+dann][+ord]`

- 图库：`clean`、`snr20`、`snr10`、`snr5`、`snr0`、`snr-5`，以及第二轮的 `snr10_n2`、`snr5_n2`、`snr0_n2`（换噪声种子）、`snr5_pink`、`snr0_pink`（粉红噪声）。
- `_few<n>`：每类只有 n 个带标签样本。
- `+mix`：多工况混合训练；`+adabn`、`+dann`：无监督域自适应；`+ord`：30-2 用阶次归一化图库。

每个场景都报告 20-0 测试集（`test`）、30-2 全部样本（`cross`）和 30-2 时间上最后 200 个样本（`xtail`）。比较跨工况方法时看 `xtail`。

第二轮实验的复现命令：

```powershell
python run_bench.py --device cuda --scenarios snr10_n2,snr5_n2,snr0_n2,snr5_pink,snr0_pink --models cnn,resnet --out results/bench2.csv
python run_bench.py --device cuda --scenarios clean,clean+adabn,clean+mix,snr5,snr5+adabn,snr5+mix --models resnet --out results/bench2.csv
```

## 用真实噪声录音

```powershell
python make_banks.py --csv_dir 东南大学数据\gearbox\gearset --out banks --noise_wav 矿区背景.wav --levels snr5,snr0 --tag _mine
python run_bench.py --device cuda --scenarios snr5_mine,snr0_mine --models cnn,resnet
```

录音需为 PCM wav，采样率与信号相同。

## 重新生成图库（一般不需要）

```powershell
python make_banks.py --csv_dir 东南大学数据\gearbox\gearset --out banks --workers 4 --check_mat ..\data\data_total.mat
```

原始数据：<https://github.com/cathysiyu/Mechanical-datasets>（gearbox/gearset，每个 CSV 约 80 MB）。论文中引用：Shao S., McAleer S., Yan R., Baldi P. Highly accurate machine fault diagnosis using deep transfer learning. *IEEE Transactions on Industrial Informatics*, 2019, 15(4): 2446–2455。

ECA 注意力参考：Wang Q., Wu B., Zhu P., et al. ECA-Net: Efficient channel attention for deep convolutional neural networks. *CVPR*, 2020。本项目的实现是自写的。
