# v28(泵,GPU):频带增益增强的端到端探索性筛查

只用 Bruinsma 泵数据(Motor-2,100% 转速)。75% 转速封存数据不打开。**这是探索性筛查,不是确认,不做多重校正。**
预注册:`v28_endtoend_PREREGISTERED.txt`(分支 `v28-prereg`,UTC 2026-11-18 14:33:36 推送,早于任何代码和运行)。

## 放哪里
`D:\DCWRU_Data\Paderborn_Experiment\`:
- **用本分支的 `m2_paderborn_kr.py` 覆盖旧文件**(新增 `--aug-band`、`--band-gain`,默认关闭;默认参数下与旧版逐位一致,已在 CPU 上核对)。`kr_modules.py` 不变,不用动。
- 新增 `run_batch_v28_cf.ps1`、`analyze_v28.py`。
- `runs_paderborn\V20_*_Bg4_seed0`(旧 Bg4,含 `feat_gcd_PADERBORN_seed0.npz`)必须还在。

## 运行
    cd D:\DCWRU_Data\Paderborn_Experiment
    .\run_batch_v28_cf.ps1
1. 先检查两个代码文件的 sha256,不对就停止。
2. **步骤 0(GPU 一致性检查):** 用新代码、默认参数在 GPU 上重跑 C1 fold 1 的 Bg4(标签 `V28CHK_C1_f1_Bg4_seed0`),H 与旧 Bg4 相差 ≤ 0.005 才复用旧 Bg4;超出(或无法评估)则自动用新代码重跑全部 9 个 Bg4(标签 `V28_*_Bg4_seed0`)。检查运行本身不进入任何比较。
3. **步骤 1:** 3 个新组 × 9 个划分 × 种子 0 = 27 个运行:BI65(独立,gain 0.65)、BC2(协调,gain 2.0)、BI2(独立,gain 2.0)。估计 1 到 1.5 小时(估计,未实测);若 Bg4 需要重跑,再加 9 个。
4. 失败或损失为 nan/inf 的运行记入 `invalid_v28.txt`,批处理继续。**不要手动重跑,不要因为 H 的高低丢弃运行。**
5. 结束后自动运行 `analyze_v28.py`:先验证配对(标注窗口、样本数、参数),不通过就停止;再按预注册规则给出 5 个比较(t 区间)、描述性指标、未知类可分性 P1 和训练中干预检查。

## 上传
`summary_v28.txt`、`summary_v28.csv`、`pairing_report_v28.txt`、`bg4_check_v28.txt`、`invalid_v28.txt`(如果有),`log_V28*.txt`(含检查运行),以及每个 `V28*` 运行目录里的 `feat_gcd_PADERBORN_seed0.npz`,和 `V20_*_Bg4_seed0` 目录里的同名文件(用于 P1)。旧 Bg4 的特征文件不在的话,P1 只在三个新组之间比较,**不会为此补跑 Bg4**。

## 新组
| 名称 | 第二视图(预热后) |
|---|---|
| BI65 | 各通道独立的频带增益,gain 0.65 |
| BC2 | 四个通道共享同一组频带增益随机数,gain 2.0 |
| BI2 | 各通道独立,gain 2.0 |
所有窗口 κ = 1(不使用 K、R)。已知局限(写在预注册里):三个 C3 划分里 BI65 的特征变化比 BC2 弱约一半,BC2 在 κ = 1 下被保真规则收缩,实际扰动比名义 2.0 弱。
