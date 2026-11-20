# 矿盾声鉴（KDSJ）v3.0 · Python 版

设备声音/振动信号的分解、时频分析与智能识别软件。v3.0 起，软件主体从 MATLAB 改为 Python。MATLAB 版（v2.3）保留在 `challenge-cup-v2.3` 分支，作为对照。

## 目录

```
kdsj/                 软件本体（申报软著的源代码）
  audio.py            wav 读取/写入（支持多声道 EXTENSIBLE 格式）、分帧
  tfr.py              同步提取变换时频图（精确版 + 长音频快速版）、对数梅尔谱、jet 颜色
  vmd.py              变分模态分解、包络熵/排列熵/峭度、参数编解码、VMD 降噪
  optim.py            TTAO、BKA、随机搜索，带评估缓存和预算
  nets.py             CNN、ResNet、ECA-ResNet
  lssvm.py            LS-SVM 一对一多分类 + BKA 超参数寻优
  recognizer.py       识别器：训练、保存、加载、预测
  monitor.py          整段音频滑窗识别、告警事件、CSV 报告
  cli.py              命令行入口
  selftest.py         自检
experiments/          实验脚本（不属于软件本体）
  mimii_bench.py      MIMII 工业设备声音识别实验
  mimii_summary.py    汇总实验结果并做配对检验
  mimii_extract.py    从 MIMII 压缩包提取第 1 通道（已在服务器上做完）
  mimii_banks.py      生成时频图库（已在服务器上做完）
results/              结果
requirements.txt
```

## 安装（Windows / PowerShell）

```powershell
python -m pip install numpy
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 版 PyTorch
python -m kdsj selftest                                                          # 应显示"0 项失败"
```

如果 `cu121` 装不上，到 <https://pytorch.org/get-started/locally/> 选择与显卡驱动匹配的 CUDA 版本。

## 运行 MIMII 实验（在本机 GPU 上）

1. 从数据分支 `challenge-cup-data-mimii` 下载，把里面的 `mimii_banks/` 文件夹放到本目录下（与 `kdsj/` 同级）。
2. 在本目录打开 PowerShell：

```powershell
cd experiments
# 先跑一组试试（几分钟）
python mimii_bench.py --banks ..\mimii_banks --device cuda --protocol file --features set --models resnet --seeds 1

# 完整实验：2 种划分 x 2 种特征 x 2 种网络 x 5 个种子 = 40 组
python mimii_bench.py --banks ..\mimii_banks --device cuda

# 汇总
python mimii_summary.py results\mimii_bench.csv --out results\mimii_summary.md
```

- 默认只用 4 个 CPU 线程，不会让笔记本卡住。网络在 GPU 上训练，BKA-LSSVM 在 CPU 上算。
- 中途关掉没关系，再运行同一条命令会跳过已完成的组。
- 每种配置第一个完成的模型会保存到 `experiments\models\`，以后界面直接加载。
- 跑完后请把 `results\mimii_bench.csv` 发给我（上传到 GitHub 任意分支即可），我来写结论。

## 实验设置

| 项目 | 内容 |
| --- | --- |
| 数据 | MIMII 0 dB 子集（已混入真实工厂背景噪声），4 种设备 × 4 台，共 18019 段 10 s 录音，取第 1 通道 |
| 任务 | 8 类：风机/水泵/滑轨/阀门 × 正常/异常 |
| 划分 `file` | 每台设备每种状态按文件编号排序，60% 训练 / 20% 验证 / 20% 测试（测试 3617 段） |
| 划分 `id` | 编号 00/02/04 的设备训练和验证，编号 06 的设备全部测试（测试 4249 段，没见过的设备） |
| 特征 | `set` 同步提取变换图，`mel` 对数梅尔谱图，都是 64×64 |
| 网络 | `cnn`（原方案）、`resnet`（v2.2 起的默认方案） |
| 训练 | Adam 1e-3，批大小 32，最多 60 轮，验证损失 8 轮不降就停，类别加权交叉熵（异常样本少） |
| LSSVM | fc1 64 维特征，BKA（种群 12、迭代 20）在训练+验证集上做 5 折交叉验证选参数。LS-SVM 训练是 O(n³)，所以交叉验证每类抽 200 个、最终训练每类抽 600 个 |
| 指标 | 8 类准确率、平衡准确率、宏 F1；告警检出率（异常被判为异常的比例）、误报率（正常被判为异常的比例） |

服务器上用 CPU 跑完了 2 组（`results/mimii_partial_server.csv`，原 CNN + 同步提取变换图）：网络自身 76–82%（检出率 62–64%，误报率 11–18%）；加 BKA-LSSVM 后 86–87%（检出率约 76%，误报率约 9%）。只有 2 次，不能下结论。

## 命令行

```powershell
python -m kdsj vmd 信号.wav --optimizer ttao                       # VMD 参数寻优与分解
python -m kdsj analyze experiments\models\file_set_resnet.pt 录音.wav --report 报告.csv   # 整段音频识别与告警
```

## 说明

- 第三方依赖与引用见 `THIRD_PARTY.md`。
- 图形界面还在做，等界面设计图。
