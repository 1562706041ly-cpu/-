# v24s2 (泵,GPU):剂量与预算分配的探索性筛查

只用 Bruinsma 泵数据(Motor-2,100% 转速)。75% 转速封存数据不打开。这是探索性筛查,结果只是候选,不是确认。

## 放哪里
`D:\DCWRU_Data\Paderborn_Experiment\`:
- **用本分支的 `m2_paderborn_kr.py` 覆盖旧文件**(新增 `--kr2-shuf-scope` 参数和日志,默认行为与 v23 相同,已在 CPU 上逐位核对)。`kr_modules.py` 不变。
- 新增 `run_batch_v24s2_cf.ps1`、`analyze_v24s2.py`。
- `runs_paderborn\V20_*_Rk_seed0` 和 `V20_*_Bg4_seed0` 必须还在(作对照)。

## 运行
    cd D:\DCWRU_Data\Paderborn_Experiment
    .\run_batch_v24s2_cf.ps1
- 先检查两个代码文件的 sha256,不对就停止。
- 4 个方案 × 9 个划分 × 种子 0 = 36 个运行,估计 40 到 70 分钟(估计,未实测)。
- 失败或损失为 nan/inf 的运行记入 `invalid_v24s2.txt`,批处理继续。**不要手动重跑**,也不要因为 H 的高低丢弃运行。

## 上传
`summary_v24s2.txt`、`summary_v24s2.csv`、`pairing_report_v24s2.txt`、`invalid_v24s2.txt`(如果有),以及 `log_V20_*_Rk*_seed0.txt`(机制日志要用)。

## 方案
| 名称 | 分配 | gain |
|---|---|---|
| Rk2 | 自适应(按 K、R) | 2 |
| Rk4 | 自适应 | 4 |
| Rks2 | 随机打乱每个窗口的预算 κ(模态预算保持自然) | 2 |
| Rks4 | 同上 | 4 |
对照:已有的 Rk(gain 1)和 Bg4。判据见 `v24s2_PREREGISTERED.txt`。
