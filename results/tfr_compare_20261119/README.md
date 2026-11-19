# 时频图对比实验（2026-11-19）

## 目的

比较三种时频图作为 CNN-BKA-LSSVM 输入时的分类效果：

| 简称 | 生成方式 |
| --- | --- |
| 原图 | 原教程 `Step1.m` + `Step2_JPG_Resize.m`：画图、截图存 JPG、再读回缩放到 64×64 |
| v2.0 新图 | `set_transform` + `tfr_to_rgb`，先对数压缩，再用 `resize_bilinear` 逐点取样缩放 |
| v2.1 新图 | 同上，但改用 `resize_area` 面积平均缩放，再对数压缩（现在的默认设置） |

## 设置

- 数据：`data_total.mat`（东南大学齿轮箱 20-0 工况，5 类 × 200 个样本，窗长 2048，步长 1000）。
- 划分：每类按时间顺序 60/20/20 划分训练/验证/测试，边界各丢 2 个样本。测试集每类 40 个，共 200 个。
- CNN：`tools/py_cnn_features.py`（PyTorch，结构与 `run_step3_cnn_lssvm.m` 相同，验证集早停）。
- LSSVM：`tools/lssvm_from_features.m`（Octave 10.3 运行 v2 的 `lssvm_stage`：BKA 种群 12、迭代 20，在训练+验证集上做 5 折交叉验证寻优，测试集只评估一次）。
- 每种图跑 10 个随机种子（1–10），三种图共用同一组种子。
- 没有用 MATLAB Deep Learning Toolbox，所以 CNN 的数值和 MATLAB 上会有小差别。

## 结果（测试集，10 次运行，均值 ± 标准差 [最小, 最大]）

| 输入 | CNN 单独 | CNN-BKA-LSSVM | macro-F1 |
| --- | --- | --- | --- |
| 原图 | 98.85% ± 0.58% [97.5, 99.5] | 99.35% ± 0.24% [99.0, 99.5] | 0.9935 |
| v2.0 新图 | 93.70% ± 2.86% [87.0, 97.0] | 97.10% ± 0.77% [96.0, 98.5] | 0.9710 |
| **v2.1 新图** | **99.45% ± 0.60%** [98.5, 100] | **99.70% ± 0.35%** [99.0, 100] | **0.9970** |

配对 Wilcoxon 符号秩检验（按种子配对）：

- v2.0 新图比原图差，CNN 和 CNN-BKA-LSSVM 都是 p = 0.002。
- v2.1 新图比原图好，但不显著：CNN p = 0.055，CNN-BKA-LSSVM p = 0.094。
- BKA-LSSVM 相对 CNN 单独：原图 p = 0.008，v2.0 新图 p = 0.006，v2.1 新图 p = 0.25（不显著）。

明细见 `lssvm_*.csv`（列：特征文件、种子、CNN 准确率、LSSVM 准确率、macro-F1、gam、sig2、交叉验证错误率、无法判别样本数）和 `cnn_log.txt`。样例图见 `tfr_samples.png`（从上到下：原图、v2.0、v2.1；从左到右：Chipped、Health、Miss、Root、Surface 各 1 张）。

## 结论

1. **v2.0 的新图有缺陷，已修复。** 同步提取变换的矩阵很稀疏（约 7.5% 非零）。v2.0 从 2048 列缩到 64 列时逐点取样，只取到约 1/32 的列，图上只剩零散的点（见 `tfr_samples.png` 第二行）。CNN 准确率因此掉到 93.7%，波动也大。改成面积平均后，图上能看到连续的频率脊线，准确率恢复到 99.45%。
2. **v2.1 新图略好于原图，但差距在测试集 1 个样本左右（200 个里错 1 个 vs 错 2 个），统计上不显著。** 答辩可以说"新时频图不依赖截图，速度快、可复现，准确率不低于原方法"，不要说"显著提升"。
3. **BKA-LSSVM 只在 CNN 特征不够好时有用。** 用原图和 v2.0 图时它能补上 0.5–3.4 个百分点；用 v2.1 图时 CNN 已经接近 100%，交叉验证错误率全部为 0，BKA 无从优化。这再次说明这套齿轮箱数据太简单，要证明 BKA-LSSVM 的价值，需要加噪声、换工况或者用矿区音频。

## 复现

```bash
# 1. Octave/MATLAB：生成 v2.1 时频图（config.m 中 cfg.tfr.resize = 'area'）
run_step2_tfr
# 2. 把 resizeimg.mat 转成 npz（X: N×64×64×3，y: 1..5）后训练 CNN 并导出 fc1 特征
python tools/py_cnn_features.py cmp/area.npz cmp/area_s1 1
# 3. Octave/MATLAB：BKA-LSSVM 阶段
addpath('tools'); lssvm_from_features('cmp/area_s1', 1, 'cmp/lssvm_area.csv')
```
