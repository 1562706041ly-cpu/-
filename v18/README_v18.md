# kr_code_v18 —— 第三章验证（只做第三章）

## 怎么跑
1. 把压缩包里的文件解压到 `D:\DCWRU_Data\Paderborn_Experiment\`，覆盖同名文件。**不要覆盖你本地真正的 `gpu_runtime.py`**（包里没有放这个文件）。
2. 两个脚本：
   - `powershell -ExecutionPolicy Bypass -File run_batch_v18_paderborn.ps1`
     给 P_B、P_A3g、P_A3d、P_A3c 各补跑种子 6–11，共 24 次，和本地已有的种子 0–5 合并统计。
   - `powershell -ExecutionPolicy Bypass -File run_batch_v18_cf.ps1`
     离心泵：4 个方法 × 27 对，共 108 次。需要 `centrifugal\splits`（来自 centrifugal_splits_data_v1.zip）和 vmdpy。
   - 两个脚本可以先后跑，也可以开两个 PowerShell 窗口同时跑（共用一块显卡会慢一些）。
   - 中断后重跑同一个脚本会自动跳过已完成的运行。
3. 跑完后上传到 GitHub：`summary_v18_paderborn_vs_A3d.txt`、`summary_v18_paderborn_vs_B.txt`、`summary_v18_paderborn_vs_A3c.txt`、`summary_v18_cf.txt`、`summary_v18_cf.csv`。

## 判断标准（事先定好，不看结果再改）
- **第三章成立**：两个数据集上 A3g 的平均 H 都高于 B 和 A3c。Paderborn 至少 8/12 个种子更好，离心泵至少 18/27 对更好。
- **能量算作贡献**：两个数据集上 A3g 都高于 A3d，比例要求同上；已知误判未知的上升不超过 0.02。
- **只满足第一条**：第三章照常写，能量写成"已知性的组成部分，消融中贡献有限"。

## v18 代码的变化
- 新增 `--e1-lam`、`--e2-fidelity`、`--e3-sens` 三个能量方案。在沙箱 CPU 上跑 Paderborn 6 个种子，按事先定的标准都没通过（记为负结果，见下表），**批量脚本里不再使用**。
- 不加这三个参数时，其余行为和 v17 一致，所以新补的种子可以和 v17 的种子 0–5 合并统计。

| 配置（沙箱 CPU，种子 0–5） | 平均 H | 已知误判未知 | 对 A3g 胜/负 |
|---|---|---|---|
| A3g | 0.831 | 0.133 | — |
| A3d | 0.775 | 0.129 | 0/3 |
| E1 伪未知能量间隔 | 0.732 | 0.400 | 0/6 |
| E1d（E1 的去能量对照） | 0.729 | 0.400 | 0/6 |
| E2 能量语义保真 | 0.838 | 0.138 | 2/0（4 个种子结果完全相同） |
| E3 能量敏感度模态预算 | 0.771 | 0.167 | 1/5 |
