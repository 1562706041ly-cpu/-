# 矿盾声鉴 · 信号分解与识别算法模块 v2.2

## 目录结构

```
config.m                 所有可调参数（只改这里）
setup_paths.m            添加路径
run_selftest.m           自检（约 1 分钟，修改代码后先跑）
run_vmd_opt.m            模块一：优化 VMD 参数并分解信号（替代 main.m / main1.m）
run_vmd_compare.m        模块一：TTAO / GLO / 随机搜索 对比实验（替代 comparison.m）
run_step1_samples.m      模块二：原始数据 -> 滑窗样本（替代 data_summary.m）
run_step2_tfr.m          模块三：样本 -> 同步提取变换时频图（替代 Step1.m、Step2_JPG_Resize.m）
run_step3_cnn_lssvm.m    模块四：ResNet/CNN 特征 + BKA 优化 LSSVM（替代 CNN_BKA_LSSVM.m）
src/                     自研函数
third_party/             第三方算法（见 THIRD_PARTY.md）
tools/                   验证用的辅助脚本（不属于软件本体）
bench/                   难场景基准测试（Python/PyTorch，见 bench/README.md）
data/                    数据
results/                 运行结果（按时间戳生成子目录）
```

## 环境

- MATLAB R2019b 及以上。模块一至三不需要任何工具箱（也能在 GNU Octave 10 中运行）。
- 模块四需要 **Deep Learning Toolbox**（R2021a 及以上，需支持 `OutputNetwork` 参数）；t-SNE 图需要 Statistics Toolbox，没有时会自动跳过。

## 使用

```matlab
run_selftest          % 1. 自检
run_vmd_opt           % 2. VMD 参数寻优与分解
run_vmd_compare       % 3. 优化器对比（20 次 × 3 种算法，约 10 分钟）
run_step1_samples     % 4. 需要原始 data.mat；已有 data_total.mat 时可跳过
run_step2_tfr         % 5. 需要 data/data_total.mat，生成 data/resizeimg.mat（约 2 分钟）
run_step3_cnn_lssvm   % 6. 需要 Deep Learning Toolbox；网络由 cfg.cls.model 选择（默认 'resnet'）
```

接入矿区音频：把 `cfg.vmd.signal_file` 改成 `.wav` 路径（采样率会自动读取）；分类数据按照 `run_step1_samples.m` 末尾的示例逐类读取。

## 结果解读要点

- `run_vmd_compare` 会给出每种算法的最优、均值、标准差、命中网格最优的比例，以及秩和检验 p 值。答辩时请使用均值±标准差，不要只报单次结果。
- `run_step3_cnn_lssvm` 报告的是 `n_repeats` 次独立实验在测试集上的均值±标准差。测试集不参与任何调参。
- 时频图对比（原图 / v2.0 / v2.1）的结果见 `results/tfr_compare_20261119/README.md`。

## 数据文件

仓库不含 `.mat` 数据。请把 `data_total.mat` 放进 `data/`，再运行 `run_step2_tfr` 生成 `resizeimg.mat`。原教程生成的旧 `resizeimg.mat` 是截图 JPG 版本，和 v2.1 不同，不要混用。

## v2.2 识别网络升级

默认网络从 v2 的 3 层 CNN 换成小型 ResNet。在加噪声的测试中，ResNet 比原 CNN 显著更好，ECA 注意力没有额外提升，跨工况所有模型都失败。数据和检验见 `results/bench_20261119/README.md`。

要用加噪声的图库在 MATLAB 里测试，把数据分支里的 `banks/` 放到 `data/` 下，然后在 `config.m` 中设置：

```matlab
cfg.cls.image_file = fullfile(cfg.data_dir, 'banks', '20_0_snr5.mat');
cfg.cls.cross_file = fullfile(cfg.data_dir, 'banks', '30_2_snr5.mat');   % 可选
```
