# 预注册附录 A1.1：`gpu_runtime.py` 的出处与基线移植回归检查

所属：`prereg-da-v1`，附录 A1 第 7 节第 3 项指定由本附录记录基线移植的回归结果。**本文件是草稿，回归检查结果一栏尚未填写。** 在回归检查通过之前不开始任何其他运行。

## 1. 为什么需要本附录
`m2_paderborn_kr.py`（`v28-gpu`，sha256 `2e2c9f19a6bbde5cc097205d740b424e419b3010320acd23b693faeb743cb3af`）从 `gpu_runtime` 导入 `DEVICE`、`FAST_AUG`、`feats_gpu`、`vectorized_aug`、`save_checkpoint`。用户本地的 `gpu_runtime.py` 已被删除，仓库其他分支里的同名文件只是 CPU 冒烟测试用的占位版本（982 字节，sha256 `f4c86e34…`），不是真实文件。

## 2. 出处
- 原文件：`D:\DCWRU_Data\Paderborn_Experiment\gpu_runtime.py`，大小 3884 字节，修改时间 2026-11-02 07:24:25 UTC（来自 pyc 头部）。
- 编译产物：仓库 `main` 分支 `__pycache__/gpu_runtime.cpython-311.pyc`（git blob `b3f91c7b`），8438 字节，sha256 `88ad881580631f2f2f1b89d7dde0c7502c131425f2ca4e25919a26b0660516db`，Python 3.11 魔数 `a70d0d0a`。
- 重建文件：`gpu_runtime_recon/gpu_runtime.py`，由 pyc 反汇编手工还原，sha256 `ea887065b94e6c07692121bbd249012d36b808ebc914283050afc9d52d075046`（3723 字节，与原文件的 3884 字节不同，见第 3 节）。
- **时间风险**：pyc 记录的是 2026-11-02 07:24:25 时的源文件。如果用户在这之后又改过 `gpu_runtime.py`，而 pyc 没有重新生成，则重建结果可能早于最终使用的版本。这一点无法从现有材料判断，由第 4 节的回归检查兜底。

## 3. 字节码核对
脚本 `gpu_runtime_recon/verify_bytecode.py`（输出 `verify_log.txt`）把重建文件编译后，逐个函数与 pyc 比较 `dis.get_instructions` 的 `(opname, argval)`，同时比较 `co_varnames`、`co_names`、`co_argcount`、`co_flags`。结果：模块体 168 条、`feats_gpu` 80 条、`vectorized_aug` 175 条、`save_checkpoint` 86 条（加三个字典推导各 18 条）、`_report_memory` 47 条、`vectorized_vmd_aug` 92 条，**全部逐条一致**，函数默认参数 `(0.1, 0.85)` 也一致。

字节码一致不能还原的内容：注释、空行、行号、换行符（CRLF 或 LF）、局部变量名以外的命名风格。这就是重建文件比原文件小 161 字节的可能原因。这些内容不影响运行行为。

需要注意的一点：字节码显示 `FAST_AUG = os.environ.get('PUMP_FAST_AUG', '0') == '1'`，即环境变量未设置时默认是 `0`（legacy），不是 `1`。`m2_paderborn_kr.py` 只导入 `FAST_AUG`，所有增强调用都直接走 `vectorized_aug`，所以它不影响训练数值，只写进检查点的 `fast_aug` 字段和启动时打印的 `augmentation=` 字样。

## 4. 回归检查要求（必须通过）
在用户的 GPU 上，使用 `gpu_runtime_recon/gpu_runtime.py`：
- 数据：v2 的 `C1_fold1`；配置：E0（不加 T_adapt）；种子 0；评估**全部 800 个目标窗口**。
- 通过条件：**H = 0.694737 ± 0.005**（与 A1 第 7 节第 3 项相同，参照 `bg4_check_v28.txt` 的 0.694737）。
- 若不通过：先核对 `splits_ch1245` 与 `splits_da_v2` 的 `X_train`、`X_val` 是否逐元素相同；相同时再怀疑 `gpu_runtime.py` 与 2026-11-02 之后的版本有差异（重点是 `vectorized_aug` 的随机数消耗顺序）。

| 项目 | 结果 |
|---|---|
| 运行日期 | （待填） |
| 数据与划分文件 sha256 | （待填） |
| H | （待填） |
| 结论 | （待填） |

## 5. 约束
重建文件只在回归检查通过后才用于 `prereg-da-v1` 的任何运行。通过后，本附录补记其 sha256 作为之后所有运行共用的 `gpu_runtime.py` 版本。

## 6. 补充：移植代码的改动与已做的检查（沙箱，CPU）
- 交付的 `m2_paderborn_kr.py`（sha256 `e86d3c0e737616eeb948e59e7941b67c1c8514c6995af7f170fd2bc991383005`）相对 `v28-gpu` 版本（`2e2c9f19…`）只增加：参数 `--eval-part {auto,T_eval,T_adapt,all}` 和 `--use-adapt`，读取 `val_part`，T_adapt 追加到无标签池，`balanced_labels` 与随机抽样限定在源域行，结果里增加 `da_v1` 块和 `openset.V.known_recall_mean`（A1-D7 次要指标）。`kr_modules.py` 不变（`7dfe0e59…`）。
- 旧划分文件（没有 `val_part`）的行为不变：`--eval-part auto` 在没有 `val_part` 时等同 `all`。
- 同一套环境（沙箱 CPU，torch 2.15.0、scikit-learn 1.9.1，`g1.py` 占位版 `gpu_runtime`）、C1_fold1 v2 划分、Bg4、种子 0：原 `v28-gpu` 代码与新代码（`--eval-part all`，不加 `--use-adapt`）的 `openset`、M0、M2、`kr_diag_val`、`known_guard`、`unk_init`、`labeled_idx` 以及 `pred_*` 数组逐项相同（H 均为 0.734）。这只说明移植没有改变 E0 的行为，不是 0.694737 的复现：0.694737 是 GPU 与用户本机库版本下的数，沙箱的库版本和随机数流不同，用重建的 `gpu_runtime.py` 在 CPU 上得到 0.665，两者都不能和 0.694737 比较。
- `yu`（无标签窗口的真实标签）只用于诊断：把 T_adapt 的标签随机打乱后重跑 E0′（短训练冒烟，40 步），`FU`、`pu`、`FV` 逐元素相同，H 相同。
- 错误输入会被拒绝：`--use-adapt` 与 `--eval-part all` / `T_adapt` 同用、`--use-adapt` 用于非 generic/plain/无 VMD/非 window 归一化的配置。

## 7. 待确认（未答复）
用户尚未说明本机的 `gpu_runtime.py` 在 2026-11-02 07:24:25 UTC 之后是否改动过。回归检查通过则此项可视为无影响；不通过时先做第 4 节的 `compare_splits.py` 核对，再回到这一项。
