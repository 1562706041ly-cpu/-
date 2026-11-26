# code-da-v1：E0 / E0′ 基线移植（预注册 prereg-da-v1，附录 A1）

解压后直接在本文件夹里运行，不需要其他文件。Python 用 `D:\chuqishiyan1\zhubao\pump_reproduction\.venv_gpu\Scripts\python.exe`，数据根目录 `D:\DCWRU_Data\Paderborn_Experiment`，划分文件目录默认 `D:\DCWRU_Data\Paderborn_Experiment\centrifugal\splits_da_v2`（脚本参数 `-Splits` 可改；如果你的路径里还有一层目录，用它指定）。

## 文件
| 文件 | 用途 |
|---|---|
| `m2_paderborn_kr.py` | 训练与评估主程序（`v28-gpu` 版本加 `--eval-part`、`--use-adapt`） |
| `kr_modules.py` | 未改动（sha256 `7dfe0e59…`） |
| `gpu_runtime.py` | 重建文件，不是用户原文件（见 `addendum_A1_1.md`） |
| `verify_bytecode.py`、`verify_log.txt`、`ref/gpu_runtime_original.cpython-311.pyc` | 重建文件与原 pyc 的逐函数字节码核对（输出 ALL IDENTICAL） |
| `run_regression_check.ps1` | 步骤 0：回归检查（E0，C1_fold1，种子 0，评估全部 800 个窗口） |
| `check_regression.py` | 判定 H = 0.694737 ± 0.005，写 `regression_result.txt` |
| `compare_splits.py` | 回归失败时核对 `splits_ch1245` 与 `splits_da_v2` 的数组是否逐元素相同 |
| `run_batch_da1.ps1` | E0 与 E0′ 各 6 个划分 × 种子 0,1,2，共 36 个运行 |
| `summarize_da1.py` | 汇总成 `summary_da1.csv`（批处理结束时自动运行） |
| `addendum_A1_1.md` | 附录 A1.1 草稿，回归结果一栏留空，待你运行后填写 |

## 运行顺序
1. 在本文件夹打开 Windows PowerShell 5.1：`powershell -ExecutionPolicy Bypass -File .\run_regression_check.ps1`
   - 通过：显示 `REGRESSION PASS`，生成 `regression_PASS.txt`。
   - 不通过：脚本自动运行 `compare_splits.py`，把 `regression_result.txt`、`log_DA1_REG_E0_C1_f1_seed0_all.txt`、`compare_splits_log.txt` 发回来，不要开始批处理。
2. 通过之后：`powershell -ExecutionPolicy Bypass -File .\run_batch_da1.ps1`
   - 没有 `regression_PASS.txt` 时批处理拒绝运行（`-SkipGate` 可跳过，仅用于调试）。
   - 已完成的运行会跳过，可以中途停止再启动。失败或出现 nan 的运行记入 `invalid_da1.txt`，批处理继续，不要手工重跑或因 H 值丢弃。
   - 脚本把 CPU 线程限制为 2（OMP/MKL/OpenBLAS/NumExpr），进程优先级设为 BelowNormal，代码没有 DataLoader 工作进程；训练在 GPU（`PUMP_DEVICE=cuda`）。

脚本启动时会检查 `m2_paderborn_kr.py`、`kr_modules.py`、`gpu_runtime.py` 的 sha256，不一致就停止：
`m2_paderborn_kr.py` e86d3c0e737616eeb948e59e7941b67c1c8514c6995af7f170fd2bc991383005；
`kr_modules.py` 7dfe0e59aed59963b03386fd78a0d00ef4966d2d776d6846112385f1b88a303f；
`gpu_runtime.py` ea887065b94e6c07692121bbd249012d36b808ebc914283050afc9d52d075046。

## 配置（A1 第 3 节）
E0 与 E0′ 都用 Bg4 命令行：`--known 0,1 --unknown 2,3,4 --steps 1500 --warmup 300 --sup-steps 600 --k-score denergy --energy-T 0.05 --unk-init kr --known-subcenters 2 --guard-expand 0.2 --label-balance-key train_installation --label-record-key train_measurement --record-key train_installation --refresh-every 300 --gate-gamma 2 --input spec --norm window --mode generic --contrast plain --vmd-bank none --known-guard 0.4 --guard-group-key train_installation`（600 步监督预训练 + 1500 步对比阶段，A1-D6）。批处理再加 `--eval-part T_eval`，E0′ 另加 `--use-adapt`。运行标签 `DA1_E0_<组合>_f<折>_seed<种子>`、`DA1_E0p_...`，结果在 `runs_paderborn\<标签>\results_exp2.json`。

类别：划分文件里标签已经统一为已知 0,1、未知 2,3,4。C1：已知 healthy + motor BPFO；C2：healthy + motor BPFI；C3：healthy + soft foot；每个组合 3 个未知类（见 A0 与预注册第 2 节）。

## 代码改动
1. `--eval-part {auto,T_eval,T_adapt,all}`：读取 `val_part`（0 = T_adapt，1 = T_eval）。`auto` 对 v2 划分文件是 `T_eval`，对没有 `val_part` 的旧文件是全部窗口（旧行为不变）。`all` 只用于回归检查。评估指标 V（`openset`、`M2.V`、`kr_diag_val`、组诊断）只在所选窗口上计算。
2. `--use-adapt`（E0′）：把 T_adapt 窗口追加到源域窗口之后，进入所有用到无标签窗口的地方：对比阶段的无标签批次、K/R 刷新、`--input spec` 的频谱标准化统计、最终特征标准化、半监督 k-means 的拟合集、`--unk-init kr` 的 K/R 计算、`--guard-expand` 的伪已知参考窗口。只允许 `--mode generic --contrast plain --vmd-bank none --norm window`，且必须 `--eval-part T_eval`，否则报错。T_eval 不参与训练、校准或阈值。T_adapt 的元数据（安装名等）从 `val_*` 字段取，缺失的字段填 `NA`。
3. 有标签窗口的抽样限定在源域行（`YTR[:N_SRC]`），E0′ 与 E0 的 `labeled_idx` 相同（已核对）。
4. `yu` 只做诊断：`kr_unknown_init` 与 `known_expand_select` 只把它写进 `*_EVAL_ONLY` 日志项。核对方法：把 T_adapt 的标签随机打乱后再跑 E0′（短训练），`FU`、`pu`、`FV` 与 H 逐元素相同。注意 E0′ 里 `M2.U`、`final_kr_diag_train_EVAL_ONLY` 等 U 侧指标现在也包含 T_adapt 窗口，只是诊断。
5. 结果里增加 `da_v1` 块（eval_part、窗口数）和 `openset.V.known_recall_mean`（A1-D7 次要指标，已知类逐类召回的平均）。

## 已做的检查与限制
- 沙箱 CPU（torch 2.15.0、scikit-learn 1.9.1）：原 `v28-gpu` 代码与本代码在 C1_fold1、Bg4、种子 0、`--eval-part all` 下输出逐项相同（H 都是 0.734，原数据 v2、占位版 `gpu_runtime`）。这证明移植没有改变 E0，**不是** 0.694737 的复现：沙箱没有 GPU，库版本不同，用重建的 `gpu_runtime.py` 在 CPU 上得到 0.665。0.694737 必须在你的 GPU 上用 `run_regression_check.ps1` 检查。
- 40 步的冒烟运行覆盖了 E0、E0′、`--eval-part all`、非法组合报错和上面的标签打乱检验，H 数值没有意义，不要引用。
- PowerShell 脚本没有在 PowerShell 里运行过（沙箱没有 pwsh），只检查了 ASCII 和语法；第一次运行请看开头的 sha256 输出。
- 未解决：你没有回答本机 `gpu_runtime.py` 在 2026-11-02 之后是否改动过。回归检查通过则无影响，见 `addendum_A1_1.md` 第 7 节。
- `FAST_AUG` 默认值在字节码里是 `'0'`，不影响训练数值，只写进检查点和启动日志。
