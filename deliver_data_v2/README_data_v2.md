# 数据交付包 v2：按 R1-merged 重新生成 9 个划分

这个包只做一件事：把目标域的 T_adapt / T_eval 拆分，从主规则 R1 改成预注册 prereg-da-v1 的决定 D2（R1-merged），并在本地重新生成划分文件。窗口池、VMD 库、对数谱都不变，不需要重新下载。

## 1. 先说结论
- 源域和目标域的窗口集合、标签、折的定义都和 v1 完全相同。只有"目标窗口属于适应集还是评估集"变了。
- 变化的窗口数（R1 → R1-merged）：C1、C2、C3 的第 1 折各变 12 个窗口（`healthy 1` 的相位跳变记录被合并）；C2、C3 的第 3 折各变 100 个窗口（`bearing bpfi 3` 的两段记录被合并）；其余划分没有变化。
- 6 个主划分的窗口数与预注册一致：

| 划分 | T_adapt 窗口 / 记录 | T_eval 窗口 / 记录 |
|---|---|---|
| C1_fold1 | 491 / 15 | 309 / 11 |
| C1_fold3 | 485 / 14 | 315 / 11 |
| C2_fold1 | 486 / 15 | 314 / 11 |
| C2_fold3 | 479 / 13 | 321 / 11 |
| C3_fold1 | 495 / 15 | 305 / 11 |
| C3_fold3 | 514 / 13 | 286 / 10 |

- 第 2 折（C1、C2、C3）同样会生成，文件里带 `split_feasible = False`，不进主结果。
- 拆分记录（9 个划分的目标窗口，每窗口一行：划分、池索引、记录、侧）的 sha256 是 `b5f108dbc301ceaa62095685f069eff939f374c2aeb2d5f1487c708029862ce7`。你本地生成的结果必须与它一致。

## 2. 文件
| 路径 | 说明 |
|---|---|
| `regen_splits_v2.ps1` | 本地一键运行：生成 9 个划分，再核对，日志写到 `<Dest>\regen_splits_v2_log.txt` |
| `tools/make_splits_da_v2.py` | 生成脚本。按 R1-merged 现算拆分，并与 `target_split_v1.json` 里 `option_R1_merged` 的结果逐窗口交叉核对，不一致就报错停止 |
| `tools/verify_splits_v2.py` | 核对脚本，共 106 项检查（窗口数、记录数、无泄漏、元数据与池一致、拆分记录的 sha256 等） |
| `tools/phase_jump_meta/Motor-2_100/*.json` | 17 个安装的相位跳变测量编号（输入） |
| `target_split/target_split_v1.json` | v1 的拆分检查结果（用于交叉核对） |
| `target_split/split_assignment_r1m_v1.tsv` | 拆分记录，7200 行 |
| `target_split/split_assignment_r1m_v1.sha256` | 预期的 sha256 |
| `addendum_A0.md` | 预注册附录 A0 |

## 3. 使用步骤
前提：v1 的合并脚本已经跑过，`<Dest>\centrifugal\pool\` 里已有窗口池、对数谱及它们的 json（你已经满足）。

1. 在 GitHub 上打开分支 `data-da-v2`，点 Code → Download ZIP，解压。
2. 打开 PowerShell，进入解压出来的 `deliver_data_v2` 文件夹，运行：

```
powershell -ExecutionPolicy Bypass -File regen_splits_v2.ps1
```

需要改默认路径时：

```
powershell -ExecutionPolicy Bypass -File regen_splits_v2.ps1 -Dest D:\DCWRU_Data\Paderborn_Experiment -Py D:\chuqishiyan1\zhubao\pump_reproduction\.venv_gpu\Scripts\python.exe
```

3. 新划分写到 `<Dest>\centrifugal\splits_da_v2\`（9 个 npz，每个约 110–119 MB，加一个 tsv）。**v1 的 `splits_da_v1` 不会被改动或删除**，但之后的实验只用 `splits_da_v2`。
4. 把 `<Dest>\regen_splits_v2_log.txt` 发回来。看到最后一行是 `RESULT: ALL OK`，并且里面的 `assignment sha256` 是上面那个值，才算通过。

如果 `.ps1` 在你那边出问题，等价的命令行写法（在 `deliver_data_v2` 文件夹里，路径按你的机器改）：

```
python tools\make_splits_da_v2.py --pool <Dest>\centrifugal\pool\pool_motor2_100_ch1245.npz --logspec <Dest>\centrifugal\pool\pool_motor2_100_ch1245_logspec.npy --out <Dest>\centrifugal\splits_da_v2
python tools\verify_splits_v2.py --splits <Dest>\centrifugal\splits_da_v2 --pool <Dest>\centrifugal\pool\pool_motor2_100_ch1245.npz
```

## 4. 与 v1 的差别（键的变化）
划分文件里的键和 v1 相同，额外的键如下，现有读取代码不用改：
| 键 | v1 | v2 |
|---|---|---|
| `split_rule` | `R1` | `R1m` |
| `val_recording` / `train_recording` | `安装名\|measurement` | `安装名\|m<分量编号>` |
| `val_part` | 0 = T_adapt，1 = T_eval | 同左，但归属按 R1-merged |
| `val_pj_measurement` | 无 | 新增：每个目标窗口的相位跳变测量编号 |
| `split_feasible` | 按 R1 判定 | 按 R1-merged 判定（仍是 6 个可行、3 个不可行） |

## 5. 沙箱里做过的验证
- 脚本在沙箱里用同一份窗口池和对数谱跑通，9 个划分的 T_adapt / T_eval 窗口数全部与预注册一致。
- 脚本里的 R1-merged 实现与 `target_split_v1.json` 里由另一份代码生成的 `option_R1_merged` 逐窗口对比：2720 个窗口，0 处不同。
- 核对脚本 106 项全部通过；故意把一个适应集窗口挪到评估集后，核对脚本报出 4 项失败（窗口数、记录跨边、名义测量编号跨边、相位跳变测量编号跨边），退出码为 1。
- 连续生成两次，拆分记录的 sha256 完全一致。
- 对比 v1：9 个划分的 `train_pool_index`、`val_pool_index`、`y_train`、`y_val` 与 v1 逐项相同。

**没有做过的**：`.ps1` 没有在 Windows 上试跑过（沙箱里没有 PowerShell）。所以请你在本地运行并把日志发回来。
