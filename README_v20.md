# v20：已知类保护校准 + 第三章"能量决定扰动方向"

## 沙箱筛选结果（离心泵，种子 0，9 个划分）
| 方案 | H | 已知误判未知 | 未知召回 | 说明 |
|---|---|---|---|---|
| B（v19 新基线） | 0.605 | 0.390 | 0.762 | |
| **Bg：保护阈值按"留一安装"校准** | **0.636** | **0.206** | 0.631 | 误判率 9/9 个划分都下降，H 8/9 更高 |
| EDd：距离能量决定扰动方向（λ=0.5） | 0.614 | 0.387 | — | 比随机方向好 7/9，比 B 好 4/9 |
| EDr：随机方向（对照） | 0.601 | 0.394 | — | |
| EDd2：纯能量方向 + 扰动加倍（事后探索） | 0.547 | 0.559 | — | 明显更差 |

## 本地怎么跑（不需要下载数据，用 v19 已有的划分和本地 VMD）
1. 把这个分支里的文件复制到 `D:\DCWRU_Data\Paderborn_Experiment\`，覆盖同名文件。
   **不要**动 `centrifugal\pool\pool_motor2_100_ch1245_vmd.json`（本地 VMD 的校验文件）。
2. 运行：
```powershell
cd D:\DCWRU_Data\Paderborn_Experiment
powershell -ExecutionPolicy Bypass -File run_batch_v20_cf.ps1
```
   共 6 种方法 × 27 = 162 次。B/Bg 每次约 30 秒，ED 每次稍慢，预计 1.5–2 小时。
   可以分批跑，比如 `-Methods B,Bg3,Bg4,Bg5` 先跑完再跑 `-Methods EDdg,EDrg`；中断后重跑会跳过已完成的。
3. 跑完上传 `summary_v20_cf.txt`、`summary_v20_cf.csv`。

## 新参数
- `--guard-group-key train_installation`：已知类保护的阈值按"留一安装"校准（只用已知类的安装编号，不用评估标签）。
- `--edir {none,dist,head,random}`、`--edir-lam`：能量梯度决定 VMD 模态幅度扰动的方向；幅度上限不变。
- `--kr2-gain`：模态预算倍数（默认 1，只在探索中用过 2）。
- 不加新参数时，Paderborn 单通道结果与 v18 逐位一致（已核对）。

判断标准见 `v20_gpu_criteria_PREREGISTERED.txt`。
