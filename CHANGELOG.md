# 更新记录

## v2.1（2026-11-19）

### 修复的错误
- **新时频图缩放丢信息**：v2.0 的 `tfr_to_rgb` 用 `resize_bilinear` 把 320×2048 的同步提取变换矩阵逐点取样缩到 64×64，只取到约 1/32 的列，图上只剩零散的点，CNN 准确率从 98.9% 降到 93.7%。新增 `src/resize_area.m`（面积平均缩放），并设为默认（`cfg.tfr.resize = 'area'`）。修复后 10 次运行 CNN 99.45%、CNN-BKA-LSSVM 99.70%。详见 `results/tfr_compare_20261119/README.md`。

### 新增
- `run_selftest.m` 第 13 项：检查面积平均缩放保持均值。
- `tools/lssvm_from_features.m`：读取 `py_cnn_features.py` 导出的特征跑 BKA-LSSVM 阶段（没有 Deep Learning Toolbox 时验证用）。
- `tools/tfr_montage.m`：把几组时频图拼成对比图。
- 删除压缩包里误带的 `octave-workspace`（Octave 出错时自动保存的工作区文件）。

## v2.0

### 修复的错误
- **测试集参与调参（数据泄露）**：原 `fun.m` 用测试集准确率作为 BKA 的适应度。现改为只在“训练+验证集”内做分层 5 折交叉验证，测试集只在最后评估一次。
- **VMD 参数前后不一致**：适应度函数里用 `fix`，最终分解时用 `round`，导致最终分解的 K 与寻优得到的 K 不同（原代码实测 8 次中有 3 次不一致）。现在统一由 `vmd_decode` 解码。
- **TTAO 步长没有按量纲缩放**：原步长固定为 `9*exp(-t/T)`，对 alpha（100–2500）几乎不起作用。现在所有优化器都在 [0,1] 归一化空间里搜索，步长按搜索范围比例设置。
- **BKA 实现错误**：随机数 r 只生成一次；柯西变异只用最后一维；每代最优的更新逻辑有误。已按论文重写为 `src/bka_optimize.m`。
- **LSSVM 二分类器个数写死为 45**：5 类一对一只需要 10 个，现在按 `nchoosek(n,2)` 计算。
- **无法判别的样本被随机赋予类别**：现在计为错误，并单独统计数量。
- **`Step1.m` 中 `axis_control(flag)` 的 `flag` 未定义**，脚本无法运行。
- TTAO 种群大小不是 3 的倍数时 `X00_fit` 被覆盖。
- `plotimf` 把最后一个 IMF 标成“残差”（VMD 没有残差项）。

### 改进
- 所有参数集中到 `config.m`；固定随机种子；结果按时间戳保存到 `results/`。
- 自研省内存 VMD（`vmd_fast.m`）：结果与原版一致（最大差异约 3e-4），实测快约 7 倍，内存约为原来的 1/500，可以处理长音频。
- `EvalCounter` 缓存相同参数的适应度，避免重复做 VMD（实测 480 次调用中只需实际计算 202 次）。
- 自研向量化 LS-SVM（`lssvm_ovo.m`）：核矩阵复用，不再依赖 LSSVMlab。
- 新增 `run_vmd_compare.m`：多次独立运行，统一评估预算，用网格穷举给出参考最优，并做 Wilcoxon 秩和检验。
- 时频图在内存中直接生成（`set_transform` + `tfr_to_rgb`），不再截图保存 JPG；做了对数压缩，图像信息更清晰。
- CNN 使用验证集早停；送入 LSSVM 的是 64 维 fc1 特征；输出精确率、召回率、F1 和混淆矩阵；重复实验后报告均值±标准差。
- 信号读取支持 csv/xlsx/mat/wav/flac，方便接入矿区音频。
- 去掉统计工具箱、图像处理工具箱和信号处理工具箱依赖（只有 CNN 部分需要 Deep Learning Toolbox）。
- 新增 `run_selftest.m`：12 项自检。
