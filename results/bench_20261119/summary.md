# 基准测试汇总

每格为准确率（%），10 个种子的均值 ± 标准差。只写模型名的列是网络自己的 softmax 输出；"+LSSVM" 列是 fc1 特征 + BKA-LSSVM。"-" 表示该场景没跑这个模型。
test = 20-0 工况时间上最后的 200 个样本；cross = 30-2 工况 1000 个样本（训练时没见过）。

## test

| 场景 | CNN（v2 基线） | CNN（v2 基线）+LSSVM | 加宽 CNN | 加宽 CNN+LSSVM | ResNet | ResNet+LSSVM | ECA-ResNet | ECA-ResNet+LSSVM | 次数 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 99.55 ± 0.69 | 99.70 ± 0.48 | - | - | 99.85 ± 0.24 | 100.00 ± 0.00 | 99.75 ± 0.42 | 99.45 ± 0.72 | 10 |
| snr10 | 95.20 ± 1.81 | 97.65 ± 0.67 | 92.25 ± 2.99 | 98.40 ± 0.57 | 97.50 ± 0.91 | 98.35 ± 0.82 | 97.75 ± 1.21 | 98.25 ± 0.82 | 10 |
| snr5 | 71.15 ± 7.76 | 81.95 ± 2.05 | 64.80 ± 11.04 | 83.30 ± 2.02 | 83.05 ± 2.45 | 85.70 ± 3.03 | 82.50 ± 2.58 | 85.90 ± 1.31 | 10 |
| snr0 | 46.60 ± 4.12 | 53.60 ± 3.46 | 43.55 ± 5.98 | 53.50 ± 2.72 | 56.80 ± 2.63 | 59.60 ± 3.13 | 55.05 ± 4.09 | 58.25 ± 4.20 | 10 |
| clean_few20 | 97.70 ± 1.46 | 97.35 ± 2.85 | - | - | 97.80 ± 0.35 | 96.70 ± 1.92 | 97.15 ± 0.91 | 97.65 ± 0.78 | 10 |
| snr10_few20 | 68.50 ± 10.47 | 79.35 ± 3.84 | - | - | 82.15 ± 6.29 | 80.55 ± 4.54 | 80.50 ± 5.54 | 81.15 ± 4.14 | 10 |

## cross

| 场景 | CNN（v2 基线） | CNN（v2 基线）+LSSVM | 加宽 CNN | 加宽 CNN+LSSVM | ResNet | ResNet+LSSVM | ECA-ResNet | ECA-ResNet+LSSVM | 次数 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 37.39 ± 5.30 | 35.95 ± 9.12 | - | - | 37.03 ± 6.68 | 34.73 ± 5.61 | 37.20 ± 5.48 | 35.15 ± 5.69 | 10 |
| snr10 | 28.41 ± 2.03 | 27.05 ± 1.99 | 28.37 ± 2.85 | 25.82 ± 1.75 | 30.68 ± 2.33 | 28.55 ± 3.32 | 29.73 ± 1.19 | 27.82 ± 1.73 | 10 |
| snr5 | 26.54 ± 2.56 | 27.28 ± 1.27 | 26.80 ± 3.97 | 27.85 ± 1.55 | 27.67 ± 3.17 | 28.34 ± 2.08 | 26.99 ± 1.64 | 28.16 ± 1.07 | 10 |
| snr0 | 22.56 ± 0.94 | 24.07 ± 1.63 | 21.95 ± 1.67 | 24.96 ± 1.09 | 25.72 ± 1.80 | 27.30 ± 1.12 | 25.01 ± 2.03 | 26.59 ± 1.30 | 10 |
| clean_few20 | 39.12 ± 4.48 | 34.53 ± 6.98 | - | - | 29.99 ± 3.07 | 27.96 ± 3.78 | 29.88 ± 4.42 | 28.41 ± 4.12 | 10 |
| snr10_few20 | 24.96 ± 4.09 | 27.01 ± 3.55 | - | - | 25.94 ± 3.03 | 26.03 ± 2.68 | 26.25 ± 3.84 | 25.48 ± 2.47 | 10 |

## 配对检验（test，按种子配对，双侧精确 Wilcoxon 符号秩检验）

| 场景 | 比较 | 输出 | 平均差（百分点） | 胜/平/负 | p |
|---|---|---|---|---|---|
| clean | ECA-ResNet vs CNN（v2 基线） | softmax | +0.20 | 3/5/2 | 0.7500 |
| clean | ECA-ResNet vs CNN（v2 基线） | +LSSVM | -0.25 | 3/3/4 | 0.5781 |
| clean | ResNet vs CNN（v2 基线） | softmax | +0.30 | 4/4/2 | 0.3750 |
| clean | ResNet vs CNN（v2 基线） | +LSSVM | +0.30 | 4/6/0 | 0.1250 |
| clean | ECA-ResNet vs ResNet | softmax | -0.10 | 3/4/3 | 0.5312 |
| clean | ECA-ResNet vs ResNet | +LSSVM | -0.55 | 0/5/5 | 0.0625 |
| clean | CNN（v2 基线）：+LSSVM vs softmax | - | +0.15 | 3/5/2 | 0.8125 |
| clean | ResNet：+LSSVM vs softmax | - | +0.15 | 3/7/0 | 0.2500 |
| clean | ECA-ResNet：+LSSVM vs softmax | - | -0.30 | 1/7/2 | 0.5000 |
| snr10 | ECA-ResNet vs CNN（v2 基线） | softmax | +2.55 | 9/0/1 | 0.0156 |
| snr10 | ECA-ResNet vs CNN（v2 基线） | +LSSVM | +0.60 | 7/1/2 | 0.1406 |
| snr10 | ResNet vs CNN（v2 基线） | softmax | +2.30 | 9/1/0 | 0.0039 |
| snr10 | ResNet vs CNN（v2 基线） | +LSSVM | +0.70 | 7/1/2 | 0.0742 |
| snr10 | ECA-ResNet vs ResNet | softmax | +0.25 | 6/0/4 | 0.3281 |
| snr10 | ECA-ResNet vs ResNet | +LSSVM | -0.10 | 3/1/6 | 0.7305 |
| snr10 | 加宽 CNN vs CNN（v2 基线） | softmax | -2.95 | 2/0/8 | 0.0664 |
| snr10 | 加宽 CNN vs CNN（v2 基线） | +LSSVM | +0.75 | 6/3/1 | 0.0469 |
| snr10 | ResNet vs 加宽 CNN | softmax | +5.25 | 9/0/1 | 0.0039 |
| snr10 | ResNet vs 加宽 CNN | +LSSVM | -0.05 | 4/2/4 | 1.0000 |
| snr10 | CNN（v2 基线）：+LSSVM vs softmax | - | +2.45 | 10/0/0 | 0.0020 |
| snr10 | 加宽 CNN：+LSSVM vs softmax | - | +6.15 | 10/0/0 | 0.0020 |
| snr10 | ResNet：+LSSVM vs softmax | - | +0.85 | 9/0/1 | 0.0098 |
| snr10 | ECA-ResNet：+LSSVM vs softmax | - | +0.50 | 3/4/3 | 0.4688 |
| snr5 | ECA-ResNet vs CNN（v2 基线） | softmax | +11.35 | 9/0/1 | 0.0039 |
| snr5 | ECA-ResNet vs CNN（v2 基线） | +LSSVM | +3.95 | 9/1/0 | 0.0039 |
| snr5 | ResNet vs CNN（v2 基线） | softmax | +11.90 | 10/0/0 | 0.0020 |
| snr5 | ResNet vs CNN（v2 基线） | +LSSVM | +3.75 | 10/0/0 | 0.0020 |
| snr5 | ECA-ResNet vs ResNet | softmax | -0.55 | 5/2/3 | 0.9922 |
| snr5 | ECA-ResNet vs ResNet | +LSSVM | +0.20 | 5/1/4 | 0.9141 |
| snr5 | 加宽 CNN vs CNN（v2 基线） | softmax | -6.35 | 4/0/6 | 0.1934 |
| snr5 | 加宽 CNN vs CNN（v2 基线） | +LSSVM | +1.35 | 7/1/2 | 0.2266 |
| snr5 | ResNet vs 加宽 CNN | softmax | +18.25 | 10/0/0 | 0.0020 |
| snr5 | ResNet vs 加宽 CNN | +LSSVM | +2.40 | 8/0/2 | 0.0801 |
| snr5 | CNN（v2 基线）：+LSSVM vs softmax | - | +10.80 | 10/0/0 | 0.0020 |
| snr5 | 加宽 CNN：+LSSVM vs softmax | - | +18.50 | 10/0/0 | 0.0020 |
| snr5 | ResNet：+LSSVM vs softmax | - | +2.65 | 9/0/1 | 0.0078 |
| snr5 | ECA-ResNet：+LSSVM vs softmax | - | +3.40 | 9/1/0 | 0.0039 |
| snr0 | ECA-ResNet vs CNN（v2 基线） | softmax | +8.45 | 9/0/1 | 0.0039 |
| snr0 | ECA-ResNet vs CNN（v2 基线） | +LSSVM | +4.65 | 7/0/3 | 0.0273 |
| snr0 | ResNet vs CNN（v2 基线） | softmax | +10.20 | 10/0/0 | 0.0020 |
| snr0 | ResNet vs CNN（v2 基线） | +LSSVM | +6.00 | 8/0/2 | 0.0098 |
| snr0 | ECA-ResNet vs ResNet | softmax | -1.75 | 4/0/6 | 0.3359 |
| snr0 | ECA-ResNet vs ResNet | +LSSVM | -1.35 | 3/0/7 | 0.2598 |
| snr0 | 加宽 CNN vs CNN（v2 基线） | softmax | -3.05 | 3/0/7 | 0.1914 |
| snr0 | 加宽 CNN vs CNN（v2 基线） | +LSSVM | -0.10 | 5/0/5 | 0.9023 |
| snr0 | ResNet vs 加宽 CNN | softmax | +13.25 | 10/0/0 | 0.0020 |
| snr0 | ResNet vs 加宽 CNN | +LSSVM | +6.10 | 8/0/2 | 0.0117 |
| snr0 | CNN（v2 基线）：+LSSVM vs softmax | - | +7.00 | 9/0/1 | 0.0059 |
| snr0 | 加宽 CNN：+LSSVM vs softmax | - | +9.95 | 9/0/1 | 0.0059 |
| snr0 | ResNet：+LSSVM vs softmax | - | +2.80 | 8/0/2 | 0.0312 |
| snr0 | ECA-ResNet：+LSSVM vs softmax | - | +3.20 | 8/0/2 | 0.0430 |
| clean_few20 | ECA-ResNet vs CNN（v2 基线） | softmax | -0.55 | 3/0/7 | 0.4824 |
| clean_few20 | ECA-ResNet vs CNN（v2 基线） | +LSSVM | +0.30 | 4/1/5 | 0.6953 |
| clean_few20 | ResNet vs CNN（v2 基线） | softmax | +0.10 | 5/1/4 | 0.7617 |
| clean_few20 | ResNet vs CNN（v2 基线） | +LSSVM | -0.65 | 3/0/7 | 0.3652 |
| clean_few20 | ECA-ResNet vs ResNet | softmax | -0.65 | 1/3/6 | 0.0625 |
| clean_few20 | ECA-ResNet vs ResNet | +LSSVM | +0.95 | 5/3/2 | 0.2031 |
| clean_few20 | CNN（v2 基线）：+LSSVM vs softmax | - | -0.35 | 3/1/6 | 0.8086 |
| clean_few20 | ResNet：+LSSVM vs softmax | - | -1.10 | 1/5/4 | 0.1875 |
| clean_few20 | ECA-ResNet：+LSSVM vs softmax | - | +0.50 | 5/4/1 | 0.2500 |
| snr10_few20 | ECA-ResNet vs CNN（v2 基线） | softmax | +12.00 | 8/1/1 | 0.0078 |
| snr10_few20 | ECA-ResNet vs CNN（v2 基线） | +LSSVM | +1.80 | 6/1/3 | 0.3789 |
| snr10_few20 | ResNet vs CNN（v2 基线） | softmax | +13.65 | 8/0/2 | 0.0195 |
| snr10_few20 | ResNet vs CNN（v2 基线） | +LSSVM | +1.20 | 4/1/5 | 0.8008 |
| snr10_few20 | ECA-ResNet vs ResNet | softmax | -1.65 | 4/0/6 | 0.6113 |
| snr10_few20 | ECA-ResNet vs ResNet | +LSSVM | +0.60 | 6/0/4 | 0.7676 |
| snr10_few20 | CNN（v2 基线）：+LSSVM vs softmax | - | +10.85 | 8/0/2 | 0.0137 |
| snr10_few20 | ResNet：+LSSVM vs softmax | - | -1.60 | 4/0/6 | 0.3750 |
| snr10_few20 | ECA-ResNet：+LSSVM vs softmax | - | +0.65 | 5/0/5 | 0.7168 |

## 模型规模与训练

| 模型 | 参数量 | 平均训练轮数 | 平均训练时间（s） |
|---|---|---|---|
| CNN（v2 基线） | 28293 | 36.0 | 57.7 |
| 加宽 CNN | 81701 | 19.8 | 129.4 |
| ResNet | 81877 | 32.3 | 174.9 |
| ECA-ResNet | 81886 | 34.8 | 201.0 |
