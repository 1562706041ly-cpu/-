# 第一步数据准备交付包 v1（Bruinsma 离心泵，Motor-2，100 % 转速）

本文件夹是七步计划中第 1 步"数据准备"的交付内容：合并校验过的多通道窗口池、预先算好的 4 通道对数谱、第三章的 VMD 分解、目标域 T_adapt / T_eval 拆分检查结果，以及在本地重建 9 个划分的脚本。

## 1. 数据从哪里来

- 窗口池 `pool_motor2_100_ch1245.npz` 的两个分卷来自分支 `v19-B`（`centrifugal/pool/`），本包里的分卷与分支上的文件逐字节相同。
- VMD 分解 `pool_motor2_100_ch1245_vmd.npy` 的 8 个分卷来自分支 `v19-VMD`，同样逐字节相同。
- 分支 `v18-centrifugal-data` 顶层是 Paderborn 轴承数据，不是离心泵数据。它的 `v19/` 子目录里有同一份泵数据的副本（git blob 与 `v19-B`、`v19-VMD` 相同），本包没有使用它。
- 75 % 转速数据是封存的最终测试集，本包里没有，准备过程中也没有打开。

## 2. 文件清单

| 路径（相对 deliver_data_v1） | 大小（字节） | 说明 |
|---|---|---|
| `README_data_v1.md` | | 本说明 |
| `merge_data_v1.ps1` | | 合并、校验并生成划分的 PowerShell 脚本 |
| `centrifugal/parts_sha256.json` | | 每个分卷的大小和 sha256，以及合并后文件的 sha256 |
| `centrifugal/pool/pool_motor2_100_ch1245.npz.part00` | 47,185,920 | 窗口池分卷 1/2 |
| `centrifugal/pool/pool_motor2_100_ch1245.npz.part01` | 42,315,518 | 窗口池分卷 2/2（合并后 89,501,438 字节） |
| `centrifugal/pool/pool_motor2_100_ch1245.json` | 550 | 窗口池清单（含 sha256） |
| `centrifugal/pool/pool_motor2_100_ch1245_logspec.npy` | 44,608,128 | 4 通道对数谱（单个文件，不分卷） |
| `centrifugal/pool/pool_motor2_100_ch1245_logspec.json` | | 对数谱清单（sha256、算法、与 torch 的差值） |
| `centrifugal/pool/pool_motor2_100_ch1245_vmd.json` | 440 | VMD 清单（含 bank_sha256） |
| `centrifugal/pool_vmd/pool_motor2_100_ch1245_vmd.npy.part00` 至 `part06` | 各 47,185,920 | VMD 分卷 1 至 7 |
| `centrifugal/pool_vmd/pool_motor2_100_ch1245_vmd.npy.part07` | 26,214,528 | VMD 分卷 8（合并后 356,515,968 字节） |
| `target_split/target_split_v1.json` | | 9 个划分的 T_adapt / T_eval 拆分（机器可读） |
| `target_split/target_split_v1.csv` | | 每个池窗口一行的拆分表 |
| `target_split/check_report_v1.md` | | 拆分可行性检查报告（中文） |
| `tools/target_split_check.py` | | 生成上面三个文件 |
| `tools/build_logspec_pool.py` | | 生成对数谱 |
| `tools/make_splits_da_v1.py` | | 生成带 T_adapt / T_eval 标记的 9 个划分文件 |
| `tools/verify_delivery.py` | | 沙箱里用的整体校验脚本 |
| `tools/phase_jump_meta/Motor-2_100/*.json` | | 17 个安装的相位跳变测量编号（拆分检查用） |

所有文件都小于 100 MB。必须下载的数据约 134 MB（窗口池两个分卷加对数谱），VMD 分卷约 357 MB。

这些文件分放在仓库 `1562706041ly-cpu/-` 的两个分支上，两边都放在顶层文件夹 `deliver_data_v1/` 里，相对路径与上表相同：

- 主分支 `data-da-v1`（[data-da-v1 分支](https://github.com/1562706041ly-cpu/-/tree/data-da-v1)）：上表中除 `centrifugal/pool_vmd/` 以外的全部文件，约 134 MB。必须下载。
- VMD 分支 `data-da-v1-vmd`（[data-da-v1-vmd 分支](https://github.com/1562706041ly-cpu/-/tree/data-da-v1-vmd)）：只有 `centrifugal/pool_vmd/` 里的 8 个 VMD 分卷，外加一份相同的 `centrifugal/parts_sha256.json`，约 357 MB。只有本地还没有正确的 VMD 文件时才需要下载（见第 3 节）。

## 3. 已有 VMD 文件时不用下载 VMD 分支

如果下面这个文件已经存在，并且 sha256 等于 `0170fc642b940d05cc53f657f4bd546f6c24c4f1426c80f4063880e7710d41f8`，就不需要 `centrifugal/pool_vmd/` 里的分卷，也就不用下载分支 `data-da-v1-vmd`：

```
D:\DCWRU_Data\Paderborn_Experiment\centrifugal\pool\pool_motor2_100_ch1245_vmd.npy
```

在 PowerShell 里检查（输出是大写，和上面的值忽略大小写比较）：

```
Get-FileHash D:\DCWRU_Data\Paderborn_Experiment\centrifugal\pool\pool_motor2_100_ch1245_vmd.npy -Algorithm SHA256
```

如果这个文件不存在，或者 sha256 不对，就要另外下载分支 `data-da-v1-vmd` 的 ZIP（见第 4 节第 2 步）。合并脚本在 VMD 文件已存在且校验正确时会直接跳过；VMD 文件不存在、分卷也没下载时，只打印黄色警告，继续往下做（生成划分不需要 VMD）。VMD 文件已存在但 sha256 不对、分卷又没下载时，脚本不会动这个文件，会用红字写出文件路径和应有的 sha256，照样生成 9 个划分，但最后报 FAILED 并返回退出码 1。这时请下载分支 `data-da-v1-vmd`、按第 4 节第 2 步放好 VMD 分卷后重跑，或者先删掉这个文件。

## 4. 使用步骤

1. 在 GitHub 上打开主分支 `data-da-v1`，点 "Code"，再点 "Download ZIP"，解压到 `D:\DCWRU_Data\Paderborn_Experiment\`。解压后会多出一个文件夹（名字类似"仓库名-data-da-v1"），`deliver_data_v1` 在它里面。下面把这个 `deliver_data_v1` 叫做主交付文件夹。
2. 只有第 3 节说需要 VMD 分卷时才做这一步：用同样的方法下载分支 `data-da-v1-vmd` 的 ZIP 并解压（名字类似"仓库名-data-da-v1-vmd"）。把其中的 `deliver_data_v1\centrifugal\pool_vmd\` 整个文件夹复制到主交付文件夹的 `centrifugal\` 下面，复制后应是 `<主交付文件夹>\centrifugal\pool_vmd\pool_motor2_100_ch1245_vmd.npy.part00` 至 `part07`。VMD 分支里的 `parts_sha256.json` 与主分支里的相同，不用复制。8 个分卷要全部放好，只放一部分时脚本会报错。
3. 打开 PowerShell，进入 `deliver_data_v1` 文件夹，例如：

```
cd "D:\DCWRU_Data\Paderborn_Experiment\<解压出的文件夹>\deliver_data_v1"
powershell -ExecutionPolicy Bypass -File merge_data_v1.ps1
```

   默认目标目录是 `D:\DCWRU_Data\Paderborn_Experiment`，默认 Python 是 `D:\chuqishiyan1\zhubao\pump_reproduction\.venv_gpu\Scripts\python.exe`（需要 numpy）。需要改时：

```
powershell -ExecutionPolicy Bypass -File merge_data_v1.ps1 -Dest D:\DCWRU_Data\Paderborn_Experiment -Py D:\chuqishiyan1\zhubao\pump_reproduction\.venv_gpu\Scripts\python.exe
```

4. 脚本依次做这些事，任何一步失败都会以红字报错并返回退出码 1：
   - 先逐个检查分卷的大小和 sha256，再按文件名顺序拼接，检查合并后的 sha256，写到 `<Dest>\centrifugal\pool\`。目标文件已存在且 sha256 正确时跳过。
   - 对 VMD 文件做同样的事（见第 3 节）。窗口池文件已存在但 sha256 不对、分卷又不在时，直接失败。
   - 复制对数谱 npy 和 json 到 `<Dest>\centrifugal\pool\`，并检查 sha256。
   - 复制 `target_split_v1.json`、`target_split_v1.csv`、`check_report_v1.md` 和本说明到 `<Dest>\centrifugal\target_split_v1\`。
   - 运行 `tools\make_splits_da_v1.py --logspec ...`，在 `<Dest>\centrifugal\splits_da_v1\` 生成 9 个划分文件 `motor2_100_ch1245_{C1,C2,C3}_fold{1,2,3}.npz`（C1、C3 每个 110,629,259 字节，C2 每个 118,530,699 字节；这些文件只在本地生成，不在分支上）。
   - 运行记录写到 `<Dest>\merge_data_v1_log.txt`（UTF-8）。

## 5. 数组和键

### 窗口池 `pool_motor2_100_ch1245.npz`

| 键 | 形状 | 类型 | 含义 |
|---|---|---|---|
| X | (2720, 4, 2048) | float32 | 原始加速度波形，通道 [1, 2, 4, 5]，17 个安装各 160 个窗口 |
| channels | (4,) | int64 | [1, 2, 4, 5] |
| sample_rate_hz | () | float64 | 20000 |
| window_size | () | int64 | 2048 |
| installation | (2720,) | str | 安装名，例如 `impeller 2` |
| fault_type | (2720,) | str | 类别名，例如 `impeller`、`motor BPFO` |
| severity, measurement, column, start | (2720,) | int64 | 严重程度；名义测量编号（= column // 5，5 个 12 s 列组成一条 60 s 记录）；列号；窗口在列内的起点（采样点） |

### 对数谱 `pool_motor2_100_ch1245_logspec.npy`

- 形状 (2720, 4, 1025)，float32，行顺序与池的 X 相同（池索引）。
- 第 k 个频点（k = 0 到 1024）的频率是 k × 20000 / 2048 Hz，分辨率 9.765625 Hz。

### VMD 分解 `pool_motor2_100_ch1245_vmd.npy`

- 形状 (2720, 4, 8, 2048)，float16，用 `np.load(..., mmap_mode='r')` 读取。第 3 维是 8 个模态，按中心频率从低到高排序。

### 划分文件 `splits_da_v1/motor2_100_ch1245_{C}_fold{k}.npz`

原有的键与 `centrifugal/make_splits_from_pool.py` 生成的文件完全相同，所以现有代码可以直接读：

| 键 | 形状 | 含义 |
|---|---|---|
| X_train, y_train | (N_train, 4, 2048), (N_train,) | 源域窗口和标签（C1、C3 为 1440 个，C2 为 1600 个） |
| X_val, y_val | (800, 4, 2048), (800,) | 目标域窗口和标签（5 个安装，每类 1 个，各 160 个窗口） |
| class_names | (5,) | 本组合使用的类别名，顺序就是标签 |
| known_labels, unknown_labels | (2,), (3,) | [0, 1] 和 [2, 3, 4] |
| channels, sample_rate_hz, window_size | | 同池文件 |
| train_pool_index, val_pool_index | | 每个窗口在池中的索引 |
| train_/val_ + installation, fault_type, severity, measurement, column, start | | 每个窗口的元数据 |

新增的键：

| 键 | 形状 / 类型 | 含义 |
|---|---|---|
| val_recording | (800,) str | 目标窗口的记录键 `安装名\|measurement` |
| val_part | (800,) int8 | 0 = T_adapt（适应阶段可用的无标签数据），1 = T_eval（只用于评估） |
| train_recording | (N_train,) str | 源域窗口的记录键 |
| split_rule | str | `R1` |
| split_feasible | bool | 该划分每类在 T_adapt 和 T_eval 两边是否都有记录 |
| X_train_logspec, X_val_logspec | (N, 4, 1025) float32 | 对应窗口的对数谱（只有加 `--logspec` 时才有） |

标签编码（每个组合不同）：0 = healthy（已知），1 = 已知故障，2 到 4 = 未知类，顺序见 class_names：

| 组合 | 1（已知故障） | 2、3、4（未知类） |
|---|---|---|
| C1 | motor BPFO | impeller, pump bearing, soft foot |
| C2 | motor BPFI | impeller, pump bearing, motor BPFO |
| C3 | soft foot | impeller, pump bearing, motor BPFI |

未知类的标签只用于评估和诊断，训练和适应时不能使用。

## 6. 对数谱和 VMD 的算法

对数谱（与 `code_keep/m2_paderborn_kr.py` 中 `--norm window` 的 `zn` 加 `log_spectrum` 相同）：

```
z = ((X - X.mean(-1, keepdims=True)) / np.maximum(X.std(-1, keepdims=True), 1e-8)).astype(np.float32)   # 每个窗口、每个通道
w = 0.5 - 0.5 * cos(2 * pi * n / 2048), n = 0..2047                                                      # 周期 Hann 窗，等于 torch.hann_window(2048, periodic=True)
S = log(|rfft(z * w)| + 1e-6)                                                                            # 1025 个频点
```

- 乘窗、FFT 和取对数在 float64 下计算，最后转成 float32。
- 与 torch 参考实现的差值（全部 2720 个窗口）：torch 用 float64 时最大绝对差 1.9e-9；torch 用 float32 时最大绝对差 0.011。后者出现在幅值很小的频点（对数值约 -7.5 或更低），原因是 float32 FFT 的舍入误差；差值大于 1e-4 的只占全部数值的 0.007 %。具体数字见 `pool_motor2_100_ch1245_logspec.json` 的 `torch_check`。
- 这个数组只对应未增强的（弱）视图。训练时现有编码器是从增强后的波形现算对数谱的，并且还会用训练窗口的统计量按通道、按频点标准化，这一步没有包含在本数组里。

VMD（第三章，`centrifugal/build_vmd_pool.py`）：输入是每窗口、每通道 z 标准化后的池 X（float64），K = 8，alpha = 300，tau = 0，DC = 0，init = 1，tol = 1e-7；模态按中心频率排序，存为 float16。8 个模态相加与 z 标准化波形的相对 L2 误差为 0.0392。

## 7. T_adapt / T_eval 拆分检查结果

详见 `target_split/check_report_v1.md`。要点：

- 规则 R1（看到计数之前固定）：每个目标安装的记录（`安装名|measurement`）按编号升序排列，排名为偶数的进 T_adapt，奇数的进 T_eval。同一记录的窗口只在一边。
- 9 个划分中 6 个可行：C1_fold1、C1_fold3、C2_fold1、C2_fold3、C3_fold1、C3_fold3。
- 3 个不可行：C1_fold2、C2_fold2、C3_fold2。原因是这三折中未知类 impeller 的目标安装是 `impeller 2`，它在池中只有 1 条记录（160 个窗口），相位跳变检测也只有 1 条，所以该类只能落在一边。
- 这 3 个划分照样生成了文件，`split_feasible = False`，`val_part` 仍按 R1 给出：`impeller 2` 的 160 个窗口全部在 T_adapt，T_eval 中没有 impeller。如果直接拿这 3 个划分做评估，T_eval 只有 4 类，与其他划分不可比。
- 报告第 8 节列出了备选做法及其计数（A 列级拆分、B 记录内按时间对半、C(i) 只放 T_eval、C(ii) 不参与评估，或者不使用第 2 折）。第 10 节列出另一种记录定义 R1-merged（把名义测量和相位跳变测量合并成连通分量，避免任何一种定义下的记录横跨两边），它同样是 6 个可行、3 个不可行。这些都只是选项，本步骤没有采用任何一个，需要你在写第二步预注册之前选定。
- 报告第 4 节还提醒：名义测量编号与相位跳变检测的编号在 11 个安装上不一致，主定义 R1 下 healthy 1 和 bearing bpfi 3 各有相位跳变记录横跨两边。
- 6 个可行划分里，故障类（标签 1 到 4）共 24 个"划分 × 类"组合，其中 19 个在 T_eval 只有 1 条记录（44 到 66 个窗口），其余 5 个有 2 条。healthy 在 T_eval 有 6 或 7 条。所以单个划分里某个故障类的 T_eval 指标只来自一条 60 s 测量，没有重复。

各划分 T_adapt / T_eval 目标窗口数（R1）：

| 划分 | 判定 | T_adapt | T_eval |
|---|---|---|---|
| C1_fold1 | 可行 | 503 | 297 |
| C1_fold2 | 不可行（impeller） | 511 | 289 |
| C1_fold3 | 可行 | 485 | 315 |
| C2_fold1 | 可行 | 498 | 302 |
| C2_fold2 | 不可行（impeller） | 532 | 268 |
| C2_fold3 | 可行 | 487 | 313 |
| C3_fold1 | 可行 | 507 | 293 |
| C3_fold2 | 不可行（impeller） | 547 | 253 |
| C3_fold3 | 可行 | 522 | 278 |

## 8. 封存数据

75 % 转速的数据是第 7 步才用的一次性测试集。本包不含这部分数据，也不含指向它的路径；准备过程中没有打开它。

## 9. 重新生成

在 `deliver_data_v1` 文件夹下，池文件合并好之后（`<池>` 指 `pool_motor2_100_ch1245.npz` 的完整路径，同目录下要有它的 json）：

```
python tools\target_split_check.py --pool <池>
python tools\build_logspec_pool.py --pool <池> --out centrifugal\pool
python tools\make_splits_da_v1.py --pool <池> --out <输出目录> --target-split target_split\target_split_v1.json --logspec centrifugal\pool\pool_motor2_100_ch1245_logspec.npy
```

- `target_split_check.py` 只读元数据，不用随机数，重跑得到逐字节相同的 json 和 csv。
- `build_logspec_pool.py` 默认会和 torch 对比（没装 torch 时跳过）。在沙箱的 CPU 上，从合并后的池重算得到的数组与交付的对数谱完全相同（最大绝对差 0）。其他机器和其他 numpy 版本没有测过，重算结果的 sha256 可能和清单不一样，这时请直接比较数值差。
- 上面第二条命令的 `--out centrifugal\pool` 会覆盖本包里交付的 `pool_motor2_100_ch1245_logspec.npy` 和 `pool_motor2_100_ch1245_logspec.json`，清单里的 sha256 也会换成重算文件的值。只想对比的话，请把 `--out` 换成一个新的空目录。
- 沙箱里的整体校验：`python tools/verify_delivery.py`（检查文件大小、分卷 sha256、合并结果、对数谱、9 个划分的各项不变量、脚本是否纯 ASCII、是否涉及 75 % 数据）。

## 10. 写第二步预注册之前需要你决定和反馈的事

1. 记录定义：名义测量（主定义 R1，column // 5）、相位跳变测量，或 R1-merged（报告第 10 节）。主定义下 healthy 1（相位跳变记录 14、15）和 bearing bpfi 3（相位跳变记录 0）的窗口横跨两边，涉及所有第 1 折和 C2、C3 的第 3 折。三种定义的可行性都是 6 个可行、3 个不可行。
2. 第 2 折的处理：报告第 8 节的 A、B、C(i)、C(ii)，或者不用第 2 折。合并脚本没有采用任何一个，第 2 折的文件只带 `split_feasible = False` 标记，后续代码需要检查这个键。
3. 按类的 T_eval 指标怎么报：见第 7 节最后一条。预注册里要写明单个划分的按类指标是单条记录的估计，或者规定按类指标只在多个折上汇总后报告。
4. 合并脚本只在 Linux 上的 PowerShell 7.4 里测过，没有在 Windows PowerShell 5.1 上跑过。第一次运行后，不管成功还是失败，请把 `<Dest>\merge_data_v1_log.txt` 发回来；如果窗口里有红字，也请一起发回。
