# v19：离心泵新基线（4 通道 + 频谱前端）+ 第三章

## 为什么改
- 旧输入（只用通道 4、2048 点波形）即使所有类别都给标签，5 类准确率也只有 0.60，健康和已知故障分不开。
- 深度编码器学到了"安装指纹"：在训练时见过的安装上，已知/未知判别 AUROC 有 0.99；换到没见过的安装，降到 0.26–0.34（比随机还差）。
- 沙箱初筛结果（种子 0，9 个划分）：

  | 输入 | H | 已知误判未知 | 新安装上已知性 AUROC |
  |---|---|---|---|
  | 旧 B：ch4 波形（你的 GPU 结果） | 0.441 | 0.421 | — |
  | 4 通道波形 | 0.575 | 0.331 | 0.628 |
  | **4 通道 + 频谱（新 B）** | **0.605** | 0.390 | 0.666 |

- 第三章的增强本来就是模拟"不同记录之间的差异"，而这个数据集上的主要瓶颈正是跨安装差异，所以在这里测第三章最合适。

## 本地怎么跑（训练用你的 GPU；VMD 已在沙箱算好，本地不用算）
```powershell
cd D:\repo                     # 换成你仓库的实际路径
git fetch origin
git checkout v18-centrifugal-data
git pull
cd v19
powershell -ExecutionPolicy Bypass -File merge_v19.ps1 -Dest D:\DCWRU_Data\Paderborn_Experiment
cd D:\DCWRU_Data\Paderborn_Experiment
powershell -ExecutionPolicy Bypass -File run_batch_v19_cf.ps1
```
- `merge_v19.ps1` 会拼回窗口池（89 MB）和 VMD 分解文件（357 MB），做 sha256 校验，把代码复制到实验目录，再生成 9 个划分（`centrifugal\splits_ch1245`，大约 700 MB，几十秒）。成功时会显示 2 行 `OK` 和最后一行 `all OK`。
- `run_batch_v19_cf.ps1` 一共 108 次 GPU 训练（4 种方法 × 27 对）。进程是低优先级，CPU 限制 4 线程。中断后重跑同一个脚本，会自动跳过已经完成的运行。
- 跑完后把 `summary_v19_cf.txt`、`summary_v19_cf.csv` 上传到 GitHub。

## 代码改动（单通道 Paderborn 的输出与 v18 逐位一致，已核对）
- `--input {wave,spec}`：`spec` 在编码器内部计算对数幅度谱。输入仍是波形，所以所有波形增强（包括 VMD）都不变。
- `--norm {window,global}`：逐窗口归一化或全局归一化，筛选结果 `window` 更好。
- 支持多通道输入，NPZ 中 X 为 (N, C, L)。
- 多通道 VMD：所有划分共用一个窗口池的 VMD 文件，形状 (N_pool, C, 8, L)，用 float16 存储，划分文件里记录对应的池索引；载入时会检查重构误差。
- 第三章多通道版本：每个通道、每个模态单独统计记录差异和保真界，每个通道都必须通过保真检查。
- 判断标准见 `v19_criteria_PREREGISTERED.txt`（跑之前写好）。
