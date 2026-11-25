# 第一步数据准备：目标域 T_adapt / T_eval 拆分可行性检查（规则 R1，v1）

本报告由 `deliver_data_v1/tools/target_split_check.py` 自动生成，只读取池文件的元数据（installation、fault_type、severity、measurement、column、start），不读取波形 X，不使用任何模型输出，不使用随机数。

## 1. 数据来源

- 数据：Bruinsma 离心泵振动数据，Motor-2，100 % 转速，通道 [1, 2, 4, 5]，采样率 20 kHz，窗长 2048。
- 池文件：`pool_motor2_100_ch1245.npz`，sha256 `4be16646c108805062fcc19c523b346cafd7fad812262a3cdfcea8418157afa4`，来自分支 `v19-B`（分卷 part00、part01 合并后校验一致）。
- 分支 `v18-centrifugal-data` 顶层是 Paderborn 轴承数据，并不是离心泵数据（它的 `v19/` 子目录里有同一份泵数据的副本，git blob 与 `v19-B` 完全相同）。本报告使用 `v19-B`。
- 75 % 转速数据是封存测试集，本步骤没有打开。
- 划分定义与 `centrifugal/make_splits_from_pool.py` 相同（TYPES、COMBOS 原样复制，第 k 折每类的目标安装为 `TYPES[t][(k-1) % len(TYPES[t])]`）。

## 2. 定义

- 安装（installation）：一次故障或健康状态的安装，池中每个安装 160 个窗口。
- 测量记录（recording）：键为 `安装名|measurement`，measurement 是池文件里的字段，等于 column // 5，即 5 个 12 s 列组成的名义 60 s 测量。不同安装的 measurement 编号会重复，所以键里必须带安装名。
- 目标域：每个划分的 val 一侧，共 5 个安装，每类正好 1 个。T_adapt 是目标域里给适应阶段用的无标签部分，T_eval 是只用于评估的部分。
- 可行（FEASIBLE）：每个目标安装至少有 2 条记录。因为每类在每折只有 1 个目标安装，这等价于"每个已知类和未知类在 T_adapt 和 T_eval 两边都至少有 1 条记录"。
- 警告（不影响可行性判定）：某类某一侧窗口数少于 16（160 的 10 %），或该类 T_adapt 窗口占比不在 [0.25, 0.75] 之内。

## 3. 规则 R1（在看到任何计数之前固定）

1. 拆分按目标安装逐个进行，只依赖该安装自己的元数据。所以同一个安装在它作为目标安装的所有划分里拆法相同。
2. 记录 = 该安装 160 个池窗口中出现的不同 measurement 值，按整数升序排列。
3. 排名 r（从 0 开始）为偶数的记录进 T_adapt，为奇数的进 T_eval。
4. 每个窗口跟随它所在记录的一侧。同一记录的窗口不会出现在两边。

英文原文（写入 json 的 rule_text）：R1: assignment per target installation, metadata only. recordings = distinct pool 'measurement' values among the installation's pool windows, sorted ascending as integers; the recording at rank r (0-based) goes to T_adapt if r is even and to T_eval if r is odd. Every window inherits the side of its recording. Recording key = installation + '|' + measurement. A split is FEASIBLE iff every target installation has >= 2 recordings (then each class has >= 1 recording on each side).

## 4. 测量编号核对（池字段 column // 5 与相位跳变检测结果）

相位跳变检测结果来自 `centrifugal/npy/Motor-2_100/<安装>.json` 的 `measurement_of_column`（已复制到 `deliver_data_v1/tools/phase_jump_meta/Motor-2_100/`），只统计池实际用到的列。

| 安装 | 池用到的列数 | 名义记录数 (column//5) | 相位跳变记录数（池用到的列） | json 中 n_measurements | 两种分组一致 | 相位跳变是名义分组的细分 |
|---|---|---|---|---|---|---|
| healthy 1 | 57 | 14 | 16 | 16 | 否 | 否 |
| healthy 2 | 59 | 12 | 19 | 19 | 否 | 是 |
| healthy 3 | 55 | 12 | 30 | 32 | 否 | 是 |
| impeller 1 | 25 | 5 | 17 | 17 | 否 | 是 |
| impeller 2 | 5 | 1 | 1 | 1 | 是 | 是 |
| impeller 3 | 15 | 3 | 3 | 3 | 是 | 是 |
| bearing pump 1 | 15 | 3 | 3 | 3 | 是 | 是 |
| bearing pump 2 | 15 | 3 | 7 | 7 | 否 | 是 |
| bearing pump 3 | 15 | 3 | 5 | 5 | 否 | 是 |
| bearing bpfo 1 | 15 | 3 | 3 | 3 | 是 | 是 |
| bearing bpfo 2 | 20 | 4 | 13 | 13 | 否 | 是 |
| bearing bpfo 3 | 20 | 4 | 15 | 15 | 否 | 是 |
| bearing bpfi 1 | 15 | 3 | 3 | 3 | 是 | 是 |
| bearing bpfi 2 | 15 | 3 | 4 | 4 | 否 | 是 |
| bearing bpfi 3 | 15 | 3 | 4 | 4 | 否 | 否 |
| soft foot 1 | 15 | 3 | 4 | 4 | 否 | 是 |
| soft foot 2 | 15 | 3 | 3 | 3 | 是 | 是 |

- 池字段 measurement 与 column // 5 在全部 2720 个窗口上一致。
- 17 个安装中有 11 个安装的两种分组不一致（需要你注意）。主定义仍然是池文件的 measurement 字段（划分文件里带的就是这个字段），没有改。相位跳变定义下的 R1 结果在第 9 节单独列出，供你决定。
- 相位跳变分组不是名义分组的细分的安装（某条相位跳变记录横跨两条名义记录）：healthy 1, bearing bpfi 3。
- 交叉检查：按主定义 R1 拆分后，有没有某条相位跳变记录的窗口同时出现在 T_adapt 和 T_eval；反过来，按相位跳变定义 R1 拆分后，有没有某条名义记录横跨两边：

| 安装 | 主定义 R1 下横跨两边的相位跳变记录 | 相位跳变定义 R1 下横跨两边的名义记录 |
|---|---|---|
| healthy 1 | pj14, pj15 | 4, 10, 11, 12 |
| healthy 2 | 无 | 8, 9, 10, 11 |
| healthy 3 | 无 | 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| impeller 1 | 无 | 1, 2, 3, 4 |
| bearing pump 2 | 无 | 1, 2 |
| bearing pump 3 | 无 | 2 |
| bearing bpfo 2 | 无 | 1, 2, 3 |
| bearing bpfo 3 | 无 | 0, 1, 2, 3 |
| bearing bpfi 2 | 无 | 2 |
| bearing bpfi 3 | pj0 | 2 |
| soft foot 1 | 无 | 2 |

- 解读：如果相位跳变检测到的才是真实的记录边界，那么上表第二列不为空的安装在主定义下，同一条真实记录的窗口会分到 T_adapt 和 T_eval 两边；第三列不为空表示改用相位跳变定义时，同一条名义 60 s 记录会被分到两边。两种定义哪个更接近真实记录，需要你根据数据采集方式判断。

## 5. 源域（训练安装）统计

### C1_fold1

| 标签 | 类别 | 已知/未知 | 源安装数 | 源安装 | 记录数（名义） | 记录数（相位跳变） | 窗口数 |
|---|---|---|---|---|---|---|---|
| 0 | healthy | 已知 | 2 | healthy 2, healthy 3 | 24 | 49 | 320 |
| 1 | motor BPFO | 已知 | 2 | bearing bpfo 2, bearing bpfo 3 | 8 | 28 | 320 |
| 2 | impeller | 未知 | 2 | impeller 2, impeller 3 | 4 | 4 | 320 |
| 3 | pump bearing | 未知 | 2 | bearing pump 2, bearing pump 3 | 6 | 12 | 320 |
| 4 | soft foot | 未知 | 1 | soft foot 2 | 3 | 3 | 160 |

### C1_fold2

| 标签 | 类别 | 已知/未知 | 源安装数 | 源安装 | 记录数（名义） | 记录数（相位跳变） | 窗口数 |
|---|---|---|---|---|---|---|---|
| 0 | healthy | 已知 | 2 | healthy 1, healthy 3 | 26 | 46 | 320 |
| 1 | motor BPFO | 已知 | 2 | bearing bpfo 1, bearing bpfo 3 | 7 | 18 | 320 |
| 2 | impeller | 未知 | 2 | impeller 1, impeller 3 | 8 | 20 | 320 |
| 3 | pump bearing | 未知 | 2 | bearing pump 1, bearing pump 3 | 6 | 8 | 320 |
| 4 | soft foot | 未知 | 1 | soft foot 1 | 3 | 4 | 160 |

### C1_fold3

| 标签 | 类别 | 已知/未知 | 源安装数 | 源安装 | 记录数（名义） | 记录数（相位跳变） | 窗口数 |
|---|---|---|---|---|---|---|---|
| 0 | healthy | 已知 | 2 | healthy 1, healthy 2 | 26 | 35 | 320 |
| 1 | motor BPFO | 已知 | 2 | bearing bpfo 1, bearing bpfo 2 | 7 | 16 | 320 |
| 2 | impeller | 未知 | 2 | impeller 1, impeller 2 | 6 | 18 | 320 |
| 3 | pump bearing | 未知 | 2 | bearing pump 1, bearing pump 2 | 6 | 10 | 320 |
| 4 | soft foot | 未知 | 1 | soft foot 2 | 3 | 3 | 160 |

### C2_fold1

| 标签 | 类别 | 已知/未知 | 源安装数 | 源安装 | 记录数（名义） | 记录数（相位跳变） | 窗口数 |
|---|---|---|---|---|---|---|---|
| 0 | healthy | 已知 | 2 | healthy 2, healthy 3 | 24 | 49 | 320 |
| 1 | motor BPFI | 已知 | 2 | bearing bpfi 2, bearing bpfi 3 | 6 | 8 | 320 |
| 2 | impeller | 未知 | 2 | impeller 2, impeller 3 | 4 | 4 | 320 |
| 3 | pump bearing | 未知 | 2 | bearing pump 2, bearing pump 3 | 6 | 12 | 320 |
| 4 | motor BPFO | 未知 | 2 | bearing bpfo 2, bearing bpfo 3 | 8 | 28 | 320 |

### C2_fold2

| 标签 | 类别 | 已知/未知 | 源安装数 | 源安装 | 记录数（名义） | 记录数（相位跳变） | 窗口数 |
|---|---|---|---|---|---|---|---|
| 0 | healthy | 已知 | 2 | healthy 1, healthy 3 | 26 | 46 | 320 |
| 1 | motor BPFI | 已知 | 2 | bearing bpfi 1, bearing bpfi 3 | 6 | 7 | 320 |
| 2 | impeller | 未知 | 2 | impeller 1, impeller 3 | 8 | 20 | 320 |
| 3 | pump bearing | 未知 | 2 | bearing pump 1, bearing pump 3 | 6 | 8 | 320 |
| 4 | motor BPFO | 未知 | 2 | bearing bpfo 1, bearing bpfo 3 | 7 | 18 | 320 |

### C2_fold3

| 标签 | 类别 | 已知/未知 | 源安装数 | 源安装 | 记录数（名义） | 记录数（相位跳变） | 窗口数 |
|---|---|---|---|---|---|---|---|
| 0 | healthy | 已知 | 2 | healthy 1, healthy 2 | 26 | 35 | 320 |
| 1 | motor BPFI | 已知 | 2 | bearing bpfi 1, bearing bpfi 2 | 6 | 7 | 320 |
| 2 | impeller | 未知 | 2 | impeller 1, impeller 2 | 6 | 18 | 320 |
| 3 | pump bearing | 未知 | 2 | bearing pump 1, bearing pump 2 | 6 | 10 | 320 |
| 4 | motor BPFO | 未知 | 2 | bearing bpfo 1, bearing bpfo 2 | 7 | 16 | 320 |

### C3_fold1

| 标签 | 类别 | 已知/未知 | 源安装数 | 源安装 | 记录数（名义） | 记录数（相位跳变） | 窗口数 |
|---|---|---|---|---|---|---|---|
| 0 | healthy | 已知 | 2 | healthy 2, healthy 3 | 24 | 49 | 320 |
| 1 | soft foot | 已知 | 1 | soft foot 2 | 3 | 3 | 160 |
| 2 | impeller | 未知 | 2 | impeller 2, impeller 3 | 4 | 4 | 320 |
| 3 | pump bearing | 未知 | 2 | bearing pump 2, bearing pump 3 | 6 | 12 | 320 |
| 4 | motor BPFI | 未知 | 2 | bearing bpfi 2, bearing bpfi 3 | 6 | 8 | 320 |

### C3_fold2

| 标签 | 类别 | 已知/未知 | 源安装数 | 源安装 | 记录数（名义） | 记录数（相位跳变） | 窗口数 |
|---|---|---|---|---|---|---|---|
| 0 | healthy | 已知 | 2 | healthy 1, healthy 3 | 26 | 46 | 320 |
| 1 | soft foot | 已知 | 1 | soft foot 1 | 3 | 4 | 160 |
| 2 | impeller | 未知 | 2 | impeller 1, impeller 3 | 8 | 20 | 320 |
| 3 | pump bearing | 未知 | 2 | bearing pump 1, bearing pump 3 | 6 | 8 | 320 |
| 4 | motor BPFI | 未知 | 2 | bearing bpfi 1, bearing bpfi 3 | 6 | 7 | 320 |

### C3_fold3

| 标签 | 类别 | 已知/未知 | 源安装数 | 源安装 | 记录数（名义） | 记录数（相位跳变） | 窗口数 |
|---|---|---|---|---|---|---|---|
| 0 | healthy | 已知 | 2 | healthy 1, healthy 2 | 26 | 35 | 320 |
| 1 | soft foot | 已知 | 1 | soft foot 2 | 3 | 3 | 160 |
| 2 | impeller | 未知 | 2 | impeller 1, impeller 2 | 6 | 18 | 320 |
| 3 | pump bearing | 未知 | 2 | bearing pump 1, bearing pump 2 | 6 | 10 | 320 |
| 4 | motor BPFI | 未知 | 2 | bearing bpfi 1, bearing bpfi 2 | 6 | 7 | 320 |

源域只有 1 个安装的类（第二步"源域留一安装验证"时，这些类在留出该安装后源域里就没有该类了）：

- C1_fold1：soft foot（未知类）
- C1_fold2：soft foot（未知类）
- C1_fold3：soft foot（未知类）
- C3_fold1：soft foot（已知类）
- C3_fold2：soft foot（已知类）
- C3_fold3：soft foot（已知类）

- soft foot 只有 2 个安装，所以第 3 折重新使用 soft foot 1 作为目标安装（与第 1 折相同）。在 C1 和 C3 中，第 1 折和第 3 折的 soft foot 目标安装都是 soft foot 1，拆分也相同（源域里的 soft foot 都是 soft foot 2）。C2 不使用 soft foot。

## 6. 目标域逐划分统计（主定义：池 measurement 字段）

### C1_fold1：FEASIBLE

| 标签 | 类别 | 已知/未知 | 目标安装 | 记录 id（窗口数，侧） | T_adapt 记录/窗口 | T_eval 记录/窗口 | T_adapt 占比 | 警告 |
|---|---|---|---|---|---|---|---|---|
| 0 | healthy | 已知 | healthy 1 | 0(13,A); 1(10,E); 2(17,A); 3(5,E); 4(15,A); 5(15,E); 6(16,A); 7(8,E); 8(10,A); 9(7,E); 10(13,A); 11(12,E); 12(12,A); 13(7,E) | 7/96 | 7/64 | 0.6000 | 无 |
| 1 | motor BPFO | 已知 | bearing bpfo 1 | 0(48,A); 1(57,E); 2(55,A) | 2/103 | 1/57 | 0.6438 | 无 |
| 2 | impeller | 未知 | impeller 1 | 0(32,A); 1(36,E); 2(28,A); 3(26,E); 4(38,A) | 3/98 | 2/62 | 0.6125 | 无 |
| 3 | pump bearing | 未知 | bearing pump 1 | 0(50,A); 1(66,E); 2(44,A) | 2/94 | 1/66 | 0.5875 | 无 |
| 4 | soft foot | 未知 | soft foot 1 | 0(51,A); 1(48,E); 2(61,A) | 2/112 | 1/48 | 0.7000 | 无 |

合计：目标窗口 800，T_adapt 503 个窗口 / 16 条记录，T_eval 297 个窗口 / 12 条记录。记录 id 后面括号里是窗口数和侧（A = T_adapt，E = T_eval）。

### C1_fold2：INFEASIBLE

| 标签 | 类别 | 已知/未知 | 目标安装 | 记录 id（窗口数，侧） | T_adapt 记录/窗口 | T_eval 记录/窗口 | T_adapt 占比 | 警告 |
|---|---|---|---|---|---|---|---|---|
| 0 | healthy | 已知 | healthy 2 | 0(11,A); 1(20,E); 2(14,A); 3(14,E); 4(14,A); 5(12,E); 6(15,A); 7(13,E); 8(10,A); 9(16,E); 10(9,A); 11(12,E) | 6/73 | 6/87 | 0.4562 | 无 |
| 1 | motor BPFO | 已知 | bearing bpfo 2 | 0(42,A); 1(38,E); 2(38,A); 3(42,E) | 2/80 | 2/80 | 0.5000 | 无 |
| 2 | impeller | 未知 | impeller 2 | 0(160,A) | 1/160 | 0/0 | 1.0000 | 某一侧窗口数 < 16；T_adapt 占比超出 [0.25, 0.75] |
| 3 | pump bearing | 未知 | bearing pump 2 | 0(57,A); 1(57,E); 2(46,A) | 2/103 | 1/57 | 0.6438 | 无 |
| 4 | soft foot | 未知 | soft foot 2 | 0(49,A); 1(65,E); 2(46,A) | 2/95 | 1/65 | 0.5938 | 无 |

合计：目标窗口 800，T_adapt 511 个窗口 / 13 条记录，T_eval 289 个窗口 / 10 条记录。记录 id 后面括号里是窗口数和侧（A = T_adapt，E = T_eval）。

### C1_fold3：FEASIBLE

| 标签 | 类别 | 已知/未知 | 目标安装 | 记录 id（窗口数，侧） | T_adapt 记录/窗口 | T_eval 记录/窗口 | T_adapt 占比 | 警告 |
|---|---|---|---|---|---|---|---|---|
| 0 | healthy | 已知 | healthy 3 | 0(13,A); 1(24,E); 2(8,A); 3(11,E); 4(13,A); 5(15,E); 6(13,A); 7(14,E); 8(16,A); 9(17,E); 10(9,A); 11(7,E) | 6/72 | 6/88 | 0.4500 | 无 |
| 1 | motor BPFO | 已知 | bearing bpfo 3 | 0(37,A); 1(40,E); 2(40,A); 3(43,E) | 2/77 | 2/83 | 0.4813 | 无 |
| 2 | impeller | 未知 | impeller 3 | 0(58,A); 1(52,E); 2(50,A) | 2/108 | 1/52 | 0.6750 | 无 |
| 3 | pump bearing | 未知 | bearing pump 3 | 0(56,A); 1(44,E); 2(60,A) | 2/116 | 1/44 | 0.7250 | 无 |
| 4 | soft foot | 未知 | soft foot 1 | 0(51,A); 1(48,E); 2(61,A) | 2/112 | 1/48 | 0.7000 | 无 |

合计：目标窗口 800，T_adapt 485 个窗口 / 14 条记录，T_eval 315 个窗口 / 11 条记录。记录 id 后面括号里是窗口数和侧（A = T_adapt，E = T_eval）。

### C2_fold1：FEASIBLE

| 标签 | 类别 | 已知/未知 | 目标安装 | 记录 id（窗口数，侧） | T_adapt 记录/窗口 | T_eval 记录/窗口 | T_adapt 占比 | 警告 |
|---|---|---|---|---|---|---|---|---|
| 0 | healthy | 已知 | healthy 1 | 0(13,A); 1(10,E); 2(17,A); 3(5,E); 4(15,A); 5(15,E); 6(16,A); 7(8,E); 8(10,A); 9(7,E); 10(13,A); 11(12,E); 12(12,A); 13(7,E) | 7/96 | 7/64 | 0.6000 | 无 |
| 1 | motor BPFI | 已知 | bearing bpfi 1 | 0(55,A); 1(53,E); 2(52,A) | 2/107 | 1/53 | 0.6687 | 无 |
| 2 | impeller | 未知 | impeller 1 | 0(32,A); 1(36,E); 2(28,A); 3(26,E); 4(38,A) | 3/98 | 2/62 | 0.6125 | 无 |
| 3 | pump bearing | 未知 | bearing pump 1 | 0(50,A); 1(66,E); 2(44,A) | 2/94 | 1/66 | 0.5875 | 无 |
| 4 | motor BPFO | 未知 | bearing bpfo 1 | 0(48,A); 1(57,E); 2(55,A) | 2/103 | 1/57 | 0.6438 | 无 |

合计：目标窗口 800，T_adapt 498 个窗口 / 16 条记录，T_eval 302 个窗口 / 12 条记录。记录 id 后面括号里是窗口数和侧（A = T_adapt，E = T_eval）。

### C2_fold2：INFEASIBLE

| 标签 | 类别 | 已知/未知 | 目标安装 | 记录 id（窗口数，侧） | T_adapt 记录/窗口 | T_eval 记录/窗口 | T_adapt 占比 | 警告 |
|---|---|---|---|---|---|---|---|---|
| 0 | healthy | 已知 | healthy 2 | 0(11,A); 1(20,E); 2(14,A); 3(14,E); 4(14,A); 5(12,E); 6(15,A); 7(13,E); 8(10,A); 9(16,E); 10(9,A); 11(12,E) | 6/73 | 6/87 | 0.4562 | 无 |
| 1 | motor BPFI | 已知 | bearing bpfi 2 | 0(54,A); 1(44,E); 2(62,A) | 2/116 | 1/44 | 0.7250 | 无 |
| 2 | impeller | 未知 | impeller 2 | 0(160,A) | 1/160 | 0/0 | 1.0000 | 某一侧窗口数 < 16；T_adapt 占比超出 [0.25, 0.75] |
| 3 | pump bearing | 未知 | bearing pump 2 | 0(57,A); 1(57,E); 2(46,A) | 2/103 | 1/57 | 0.6438 | 无 |
| 4 | motor BPFO | 未知 | bearing bpfo 2 | 0(42,A); 1(38,E); 2(38,A); 3(42,E) | 2/80 | 2/80 | 0.5000 | 无 |

合计：目标窗口 800，T_adapt 532 个窗口 / 13 条记录，T_eval 268 个窗口 / 10 条记录。记录 id 后面括号里是窗口数和侧（A = T_adapt，E = T_eval）。

### C2_fold3：FEASIBLE

| 标签 | 类别 | 已知/未知 | 目标安装 | 记录 id（窗口数，侧） | T_adapt 记录/窗口 | T_eval 记录/窗口 | T_adapt 占比 | 警告 |
|---|---|---|---|---|---|---|---|---|
| 0 | healthy | 已知 | healthy 3 | 0(13,A); 1(24,E); 2(8,A); 3(11,E); 4(13,A); 5(15,E); 6(13,A); 7(14,E); 8(16,A); 9(17,E); 10(9,A); 11(7,E) | 6/72 | 6/88 | 0.4500 | 无 |
| 1 | motor BPFI | 已知 | bearing bpfi 3 | 0(60,A); 1(46,E); 2(54,A) | 2/114 | 1/46 | 0.7125 | 无 |
| 2 | impeller | 未知 | impeller 3 | 0(58,A); 1(52,E); 2(50,A) | 2/108 | 1/52 | 0.6750 | 无 |
| 3 | pump bearing | 未知 | bearing pump 3 | 0(56,A); 1(44,E); 2(60,A) | 2/116 | 1/44 | 0.7250 | 无 |
| 4 | motor BPFO | 未知 | bearing bpfo 3 | 0(37,A); 1(40,E); 2(40,A); 3(43,E) | 2/77 | 2/83 | 0.4813 | 无 |

合计：目标窗口 800，T_adapt 487 个窗口 / 14 条记录，T_eval 313 个窗口 / 11 条记录。记录 id 后面括号里是窗口数和侧（A = T_adapt，E = T_eval）。

### C3_fold1：FEASIBLE

| 标签 | 类别 | 已知/未知 | 目标安装 | 记录 id（窗口数，侧） | T_adapt 记录/窗口 | T_eval 记录/窗口 | T_adapt 占比 | 警告 |
|---|---|---|---|---|---|---|---|---|
| 0 | healthy | 已知 | healthy 1 | 0(13,A); 1(10,E); 2(17,A); 3(5,E); 4(15,A); 5(15,E); 6(16,A); 7(8,E); 8(10,A); 9(7,E); 10(13,A); 11(12,E); 12(12,A); 13(7,E) | 7/96 | 7/64 | 0.6000 | 无 |
| 1 | soft foot | 已知 | soft foot 1 | 0(51,A); 1(48,E); 2(61,A) | 2/112 | 1/48 | 0.7000 | 无 |
| 2 | impeller | 未知 | impeller 1 | 0(32,A); 1(36,E); 2(28,A); 3(26,E); 4(38,A) | 3/98 | 2/62 | 0.6125 | 无 |
| 3 | pump bearing | 未知 | bearing pump 1 | 0(50,A); 1(66,E); 2(44,A) | 2/94 | 1/66 | 0.5875 | 无 |
| 4 | motor BPFI | 未知 | bearing bpfi 1 | 0(55,A); 1(53,E); 2(52,A) | 2/107 | 1/53 | 0.6687 | 无 |

合计：目标窗口 800，T_adapt 507 个窗口 / 16 条记录，T_eval 293 个窗口 / 12 条记录。记录 id 后面括号里是窗口数和侧（A = T_adapt，E = T_eval）。

### C3_fold2：INFEASIBLE

| 标签 | 类别 | 已知/未知 | 目标安装 | 记录 id（窗口数，侧） | T_adapt 记录/窗口 | T_eval 记录/窗口 | T_adapt 占比 | 警告 |
|---|---|---|---|---|---|---|---|---|
| 0 | healthy | 已知 | healthy 2 | 0(11,A); 1(20,E); 2(14,A); 3(14,E); 4(14,A); 5(12,E); 6(15,A); 7(13,E); 8(10,A); 9(16,E); 10(9,A); 11(12,E) | 6/73 | 6/87 | 0.4562 | 无 |
| 1 | soft foot | 已知 | soft foot 2 | 0(49,A); 1(65,E); 2(46,A) | 2/95 | 1/65 | 0.5938 | 无 |
| 2 | impeller | 未知 | impeller 2 | 0(160,A) | 1/160 | 0/0 | 1.0000 | 某一侧窗口数 < 16；T_adapt 占比超出 [0.25, 0.75] |
| 3 | pump bearing | 未知 | bearing pump 2 | 0(57,A); 1(57,E); 2(46,A) | 2/103 | 1/57 | 0.6438 | 无 |
| 4 | motor BPFI | 未知 | bearing bpfi 2 | 0(54,A); 1(44,E); 2(62,A) | 2/116 | 1/44 | 0.7250 | 无 |

合计：目标窗口 800，T_adapt 547 个窗口 / 13 条记录，T_eval 253 个窗口 / 9 条记录。记录 id 后面括号里是窗口数和侧（A = T_adapt，E = T_eval）。

### C3_fold3：FEASIBLE

| 标签 | 类别 | 已知/未知 | 目标安装 | 记录 id（窗口数，侧） | T_adapt 记录/窗口 | T_eval 记录/窗口 | T_adapt 占比 | 警告 |
|---|---|---|---|---|---|---|---|---|
| 0 | healthy | 已知 | healthy 3 | 0(13,A); 1(24,E); 2(8,A); 3(11,E); 4(13,A); 5(15,E); 6(13,A); 7(14,E); 8(16,A); 9(17,E); 10(9,A); 11(7,E) | 6/72 | 6/88 | 0.4500 | 无 |
| 1 | soft foot | 已知 | soft foot 1 | 0(51,A); 1(48,E); 2(61,A) | 2/112 | 1/48 | 0.7000 | 无 |
| 2 | impeller | 未知 | impeller 3 | 0(58,A); 1(52,E); 2(50,A) | 2/108 | 1/52 | 0.6750 | 无 |
| 3 | pump bearing | 未知 | bearing pump 3 | 0(56,A); 1(44,E); 2(60,A) | 2/116 | 1/44 | 0.7250 | 无 |
| 4 | motor BPFI | 未知 | bearing bpfi 3 | 0(60,A); 1(46,E); 2(54,A) | 2/114 | 1/46 | 0.7125 | 无 |

合计：目标窗口 800，T_adapt 522 个窗口 / 14 条记录，T_eval 278 个窗口 / 10 条记录。记录 id 后面括号里是窗口数和侧（A = T_adapt，E = T_eval）。

## 7. 九个划分的判定汇总

| 划分 | 判定（主定义） | 不满足的类 | T_adapt 窗口 | T_eval 窗口 | T_adapt 记录 | T_eval 记录 | 有警告的类 | 判定（相位跳变定义） |
|---|---|---|---|---|---|---|---|---|
| C1_fold1 | FEASIBLE | 无 | 503 | 297 | 16 | 12 | 无 | FEASIBLE |
| C1_fold2 | INFEASIBLE | impeller | 511 | 289 | 13 | 10 | impeller | INFEASIBLE |
| C1_fold3 | FEASIBLE | 无 | 485 | 315 | 14 | 11 | 无 | FEASIBLE |
| C2_fold1 | FEASIBLE | 无 | 498 | 302 | 16 | 12 | 无 | FEASIBLE |
| C2_fold2 | INFEASIBLE | impeller | 532 | 268 | 13 | 10 | impeller | INFEASIBLE |
| C2_fold3 | FEASIBLE | 无 | 487 | 313 | 14 | 11 | 无 | FEASIBLE |
| C3_fold1 | FEASIBLE | 无 | 507 | 293 | 16 | 12 | 无 | FEASIBLE |
| C3_fold2 | INFEASIBLE | impeller | 547 | 253 | 13 | 9 | impeller | INFEASIBLE |
| C3_fold3 | FEASIBLE | 无 | 522 | 278 | 14 | 10 | 无 | FEASIBLE |

结论：9 个划分中 3 个不可行：C1_fold2, C2_fold2, C3_fold2。其余 6 个可行。

## 8. 不可行划分与备选方案（需要你在写预注册之前选定）

### 目标安装 impeller 2（类别 impeller），受影响的划分：C1_fold2, C2_fold2, C3_fold2

- 原因：impeller 2 在池中只有 1 条名义记录（160 个窗口），相位跳变定义下也只有 1 条，所以按记录拆分时该类只能落在一边。
- 在 C1_fold2, C2_fold2, C3_fold2 中，impeller 都是未知类。

| 方案 | 做法 | 该类 T_adapt 窗口 | 该类 T_eval 窗口 | 失去的独立性 |
|---|---|---|---|---|
| A 列级拆分 | 键为"安装名 + 竖线 + 列号"（json 中写作 installation\|c&lt;column&gt;），对排序后的列号用同样的偶数/奇数规则：c0(33,A); c1(24,E); c2(33,A); c3(37,E); c4(33,A) | 99 | 61 | 两边来自同一条连续记录中相邻的 12 s 列（同一次安装、同一次运行），只隔一个列边界 |
| B 记录内按时间对半 | 按 (列, 起点) 排序，前一半进 T_adapt，后一半进 T_eval；分界在第 2 列内部（T_adapt 最后一个窗口起点 153600，T_eval 第一个窗口起点 169984，两窗口之间相隔 14336 个采样点，约 0.72 s）；两边都有的列：2 | 80 | 80 | 两边来自同一条记录的同一个 60 s 测量；分界两侧的窗口在时间上相隔不到 1 s（见做法一栏） |
| C(i) 只放 T_eval | 该类全部窗口进 T_eval，T_adapt 中没有该类 | 0 | 160 | 评估仍与适应数据按记录独立，但适应阶段在目标域看不到该类，"每类两边都有"的条件对该类不成立 |
| C(ii) 不参与评估 | 该类全部窗口只进 T_adapt，从 T_eval 中排除 | 160 | 0 | 该划分中没有这个未知类的评估指标，T_eval 只剩 4 类，与其他划分不可直接比较 |

各方案下受影响划分的目标窗口合计（T_adapt / T_eval）：

| 划分 | A | B | C(i) | C(ii) |
|---|---|---|---|---|
| C1_fold2 | 450 / 350 | 431 / 369 | 351 / 449 | 511 / 289 |
| C2_fold2 | 471 / 329 | 452 / 348 | 372 / 428 | 532 / 268 |
| C3_fold2 | 486 / 314 | 467 / 333 | 387 / 413 | 547 / 253 |

以上方案只是供选择的选项，脚本没有采用其中任何一个，R1 的主规则也没有改。另一种做法是不使用这几个划分的第 2 折（只报告可行的划分）。选定之后再写入第二步的预注册。

## 9. 相位跳变定义下的 R1 结果（对照，不是主定义）

把记录定义换成 `安装|pj<measurement_of_column[列]>`，其余规则不变：

| 划分 | 判定 | 不满足的类 | T_adapt 窗口 | T_eval 窗口 | T_adapt 记录 | T_eval 记录 |
|---|---|---|---|---|---|---|
| C1_fold1 | FEASIBLE | 无 | 489 | 311 | 23 | 20 |
| C1_fold2 | INFEASIBLE | impeller | 549 | 251 | 24 | 19 |
| C1_fold3 | FEASIBLE | 无 | 485 | 315 | 30 | 27 |
| C2_fold1 | FEASIBLE | 无 | 495 | 305 | 23 | 19 |
| C2_fold2 | INFEASIBLE | impeller | 540 | 260 | 24 | 20 |
| C2_fold3 | FEASIBLE | 无 | 523 | 277 | 30 | 27 |
| C3_fold1 | FEASIBLE | 无 | 493 | 307 | 23 | 20 |
| C3_fold2 | INFEASIBLE | impeller | 528 | 272 | 19 | 15 |
| C3_fold3 | FEASIBLE | 无 | 525 | 275 | 24 | 22 |

各目标安装在两种定义下的 R1 计数：

| 安装 | 名义记录数 | 名义 T_adapt/T_eval 窗口 | 相位跳变记录数 | 相位跳变 T_adapt/T_eval 窗口 |
|---|---|---|---|---|
| healthy 1 | 14 | 96/64 | 16 | 87/73 |
| healthy 2 | 12 | 73/87 | 19 | 77/83 |
| healthy 3 | 12 | 72/88 | 30 | 86/74 |
| impeller 1 | 5 | 98/62 | 17 | 104/56 |
| impeller 2 | 1 | 160/0 | 1 | 160/0 |
| impeller 3 | 3 | 108/52 | 3 | 108/52 |
| bearing pump 1 | 3 | 94/66 | 3 | 94/66 |
| bearing pump 2 | 3 | 103/57 | 7 | 110/50 |
| bearing pump 3 | 3 | 116/44 | 5 | 91/69 |
| bearing bpfo 1 | 3 | 103/57 | 3 | 103/57 |
| bearing bpfo 2 | 4 | 80/80 | 13 | 107/53 |
| bearing bpfo 3 | 4 | 77/83 | 15 | 99/61 |
| bearing bpfi 1 | 3 | 107/53 | 3 | 107/53 |
| bearing bpfi 2 | 3 | 116/44 | 4 | 86/74 |
| bearing bpfi 3 | 3 | 114/46 | 4 | 139/21 |
| soft foot 1 | 3 | 112/48 | 4 | 101/59 |
| soft foot 2 | 3 | 95/65 | 3 | 95/65 |

## 10. 备选记录定义 R1-merged（只是选项，没有采用）

第 4 节显示两种记录定义（名义 column // 5 与相位跳变）在 11 个安装上不一致，主定义 R1 下有相位跳变记录横跨两边。R1-merged 把两种定义合并：在每个安装内，把池用到的列看成节点，两列只要名义 measurement 相同或相位跳变 measurement_of_column 相同就连一条边，取连通分量作为记录，键为 `安装名|m<分量编号>`，分量按其最小列号排序编号，然后用与 R1 相同的偶数/奇数规则。这样无论按哪种定义，同一条记录的窗口都不会分到两边。代价是记录更粗，记录数比名义定义少的安装：healthy 1（14 条变为 12 条）, bearing bpfi 3（3 条变为 2 条）。impeller 2 在两种定义下都只有 1 条记录，所以 R1-merged 不能解决第 2 折的问题。

这一节只是给你的选项。主规则仍是 R1，json 中主规则的条目没有变化，R1-merged 的结果单独放在 json 的 `option_R1_merged` 键下。

| 安装 | R1-merged 记录数 | 各记录（列号，窗口数，侧） | T_adapt/T_eval 窗口 |
|---|---|---|---|
| healthy 1 | 12 | m0[c0..c4, 5 列](13,A); m1[c5..c9, 4 列](10,E); m2[c11..c13, 3 列](17,A); m3[c16..c19, 3 列](5,E); m4[c20..c24, 5 列](15,A); m5[c25..c29, 5 列](15,E); m6[c30..c34, 5 列](16,A); m7[c35..c38, 3 列](8,E); m8[c41..c44, 4 列](10,A); m9[c46..c49, 3 列](7,E); m10[c50..c53, 4 列](13,A); m11[c55..c67, 13 列](31,E) | 84/76 |
| healthy 2 | 12 | m0[c0..c4, 5 列](11,A); m1[c5..c9, 5 列](20,E); m2[c10..c14, 5 列](14,A); m3[c15..c19, 5 列](14,E); m4[c20..c24, 5 列](14,A); m5[c25..c29, 5 列](12,E); m6[c30..c34, 5 列](15,A); m7[c35..c39, 5 列](13,E); m8[c40..c44, 5 列](10,A); m9[c45..c49, 4 列](16,E); m10[c50..c54, 5 列](9,A); m11[c55..c59, 5 列](12,E) | 73/87 |
| healthy 3 | 12 | m0[c0..c4, 5 列](13,A); m1[c5..c9, 5 列](24,E); m2[c10..c14, 4 列](8,A); m3[c16..c19, 4 列](11,E); m4[c20..c24, 5 列](13,A); m5[c25..c29, 5 列](15,E); m6[c30..c34, 5 列](13,A); m7[c35..c39, 5 列](14,E); m8[c40..c44, 5 列](16,A); m9[c45..c49, 5 列](17,E); m10[c50..c54, 3 列](9,A); m11[c55..c59, 4 列](7,E) | 72/88 |
| impeller 1 | 5 | m0[c0..c4, 5 列](32,A); m1[c5..c9, 5 列](36,E); m2[c10..c14, 5 列](28,A); m3[c15..c19, 5 列](26,E); m4[c20..c24, 5 列](38,A) | 98/62 |
| impeller 2 | 1 | m0[c0..c4, 5 列](160,A) | 160/0 |
| impeller 3 | 3 | m0[c0..c4, 5 列](58,A); m1[c5..c9, 5 列](52,E); m2[c10..c14, 5 列](50,A) | 108/52 |
| bearing pump 1 | 3 | m0[c0..c4, 5 列](50,A); m1[c5..c9, 5 列](66,E); m2[c10..c14, 5 列](44,A) | 94/66 |
| bearing pump 2 | 3 | m0[c0..c4, 5 列](57,A); m1[c5..c9, 5 列](57,E); m2[c10..c14, 5 列](46,A) | 103/57 |
| bearing pump 3 | 3 | m0[c0..c4, 5 列](56,A); m1[c5..c9, 5 列](44,E); m2[c10..c14, 5 列](60,A) | 116/44 |
| bearing bpfo 1 | 3 | m0[c0..c4, 5 列](48,A); m1[c5..c9, 5 列](57,E); m2[c10..c14, 5 列](55,A) | 103/57 |
| bearing bpfo 2 | 4 | m0[c0..c4, 5 列](42,A); m1[c5..c9, 5 列](38,E); m2[c10..c14, 5 列](38,A); m3[c15..c19, 5 列](42,E) | 80/80 |
| bearing bpfo 3 | 4 | m0[c0..c4, 5 列](37,A); m1[c5..c9, 5 列](40,E); m2[c10..c14, 5 列](40,A); m3[c15..c19, 5 列](43,E) | 77/83 |
| bearing bpfi 1 | 3 | m0[c0..c4, 5 列](55,A); m1[c5..c9, 5 列](53,E); m2[c10..c14, 5 列](52,A) | 107/53 |
| bearing bpfi 2 | 3 | m0[c0..c4, 5 列](54,A); m1[c5..c9, 5 列](44,E); m2[c10..c14, 5 列](62,A) | 116/44 |
| bearing bpfi 3 | 2 | m0[c0..c9, 10 列](106,A); m1[c10..c14, 5 列](54,E) | 106/54 |
| soft foot 1 | 3 | m0[c0..c4, 5 列](51,A); m1[c5..c9, 5 列](48,E); m2[c10..c14, 5 列](61,A) | 112/48 |
| soft foot 2 | 3 | m0[c0..c4, 5 列](49,A); m1[c5..c9, 5 列](65,E); m2[c10..c14, 5 列](46,A) | 95/65 |

（`c起..c止` 表示该分量的最小和最大列号，分量内的列不一定连续，完整列表见 json。）

| 划分 | R1-merged 判定 | 不满足的类 | T_adapt 窗口 | T_eval 窗口 | T_adapt 记录 | T_eval 记录 | 有警告的类 |
|---|---|---|---|---|---|---|---|
| C1_fold1 | FEASIBLE | 无 | 491 | 309 | 15 | 11 | 无 |
| C1_fold2 | INFEASIBLE | impeller | 511 | 289 | 13 | 10 | impeller |
| C1_fold3 | FEASIBLE | 无 | 485 | 315 | 14 | 11 | 无 |
| C2_fold1 | FEASIBLE | 无 | 486 | 314 | 15 | 11 | 无 |
| C2_fold2 | INFEASIBLE | impeller | 532 | 268 | 13 | 10 | impeller |
| C2_fold3 | FEASIBLE | 无 | 479 | 321 | 13 | 11 | 无 |
| C3_fold1 | FEASIBLE | 无 | 495 | 305 | 15 | 11 | 无 |
| C3_fold2 | INFEASIBLE | impeller | 547 | 253 | 13 | 9 | impeller |
| C3_fold3 | FEASIBLE | 无 | 514 | 286 | 13 | 10 | 无 |

各划分每类计数（T_adapt 记录/窗口，T_eval 记录/窗口）：

| 划分 | 标签 0 | 标签 1 | 标签 2 | 标签 3 | 标签 4 |
|---|---|---|---|---|---|
| C1_fold1 | healthy（healthy 1）：6/84，6/76 | motor BPFO（bearing bpfo 1）：2/103，1/57 | impeller（impeller 1）：3/98，2/62 | pump bearing（bearing pump 1）：2/94，1/66 | soft foot（soft foot 1）：2/112，1/48 |
| C1_fold2 | healthy（healthy 2）：6/73，6/87 | motor BPFO（bearing bpfo 2）：2/80，2/80 | impeller（impeller 2）：1/160，0/0 | pump bearing（bearing pump 2）：2/103，1/57 | soft foot（soft foot 2）：2/95，1/65 |
| C1_fold3 | healthy（healthy 3）：6/72，6/88 | motor BPFO（bearing bpfo 3）：2/77，2/83 | impeller（impeller 3）：2/108，1/52 | pump bearing（bearing pump 3）：2/116，1/44 | soft foot（soft foot 1）：2/112，1/48 |
| C2_fold1 | healthy（healthy 1）：6/84，6/76 | motor BPFI（bearing bpfi 1）：2/107，1/53 | impeller（impeller 1）：3/98，2/62 | pump bearing（bearing pump 1）：2/94，1/66 | motor BPFO（bearing bpfo 1）：2/103，1/57 |
| C2_fold2 | healthy（healthy 2）：6/73，6/87 | motor BPFI（bearing bpfi 2）：2/116，1/44 | impeller（impeller 2）：1/160，0/0 | pump bearing（bearing pump 2）：2/103，1/57 | motor BPFO（bearing bpfo 2）：2/80，2/80 |
| C2_fold3 | healthy（healthy 3）：6/72，6/88 | motor BPFI（bearing bpfi 3）：1/106，1/54 | impeller（impeller 3）：2/108，1/52 | pump bearing（bearing pump 3）：2/116，1/44 | motor BPFO（bearing bpfo 3）：2/77，2/83 |
| C3_fold1 | healthy（healthy 1）：6/84，6/76 | soft foot（soft foot 1）：2/112，1/48 | impeller（impeller 1）：3/98，2/62 | pump bearing（bearing pump 1）：2/94，1/66 | motor BPFI（bearing bpfi 1）：2/107，1/53 |
| C3_fold2 | healthy（healthy 2）：6/73，6/87 | soft foot（soft foot 2）：2/95，1/65 | impeller（impeller 2）：1/160，0/0 | pump bearing（bearing pump 2）：2/103，1/57 | motor BPFI（bearing bpfi 2）：2/116，1/44 |
| C3_fold3 | healthy（healthy 3）：6/72，6/88 | soft foot（soft foot 1）：2/112，1/48 | impeller（impeller 3）：2/108，1/52 | pump bearing（bearing pump 3）：2/116，1/44 | motor BPFI（bearing bpfi 3）：1/106，1/54 |

R1-merged 下 9 个划分中 3 个不可行：C1_fold2, C2_fold2, C3_fold2。

## 11. 输出文件与复现命令

- `target_split_v1.json`：规则文本、规则版本 R1、池 sha256、每个划分的目标池索引、每个窗口的记录键和侧、每类计数、判定、逐安装拆分表、备选方案计数。
- `target_split_v1.csv`：所有曾作为目标窗口的池窗口，每行一个（pool_index, installation, fault_type, measurement, column, start, recording_key, side）。同一安装在所有划分中拆法相同，所以每个窗口只有一行。
- `check_report_v1.md`：本报告。

沙箱中的复现命令：

```
/projects/sandbox/.venv/bin/python /projects/sandbox/deliver_data_v1/tools/target_split_check.py --pool /projects/sandbox/.scratch_data/pool/pool_motor2_100_ch1245.npz
```

本地（Windows）在 deliver_data_v1 目录下，合并好池文件后：

```
python tools\target_split_check.py --pool <池文件路径>\pool_motor2_100_ch1245.npz
```

脚本不使用随机数，连续运行两次得到的 json 和 csv 字节完全相同。
