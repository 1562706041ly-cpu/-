# 框架图（draw.io 可编辑）

- `framework_etf_gcd.drawio`：框架总图，可在 draw.io 中直接编辑。
- `make_drawio.py`：生成这张图的脚本。改了设计后可以重新运行 `python make_drawio.py` 再生成；如果你已经在 draw.io 里手动调整过，就不要再运行，否则会覆盖。

## 打开方法
1. 在网页打开 https://app.diagrams.net
2. 选 **File → Open from → Device**，选择 `framework_etf_gcd.drawio`。
   也可以用 **Open from → GitHub / URL**，直接填本分支里这个文件的 raw 地址。
3. 编辑后可导出为 PNG、PDF、SVG。

## 图的结构
- **A 数据准备**：源域（训练安装，10% 标签）、目标域（新设备，全部无标签，按测量记录分成 T_adapt / T_eval）、预处理（4 通道对数谱；对照：时频图）、弱视图、第三章模块产生的强视图。
- **B 阶段 ① 预训练**：BYOL（在线 / 目标网络、EMA、stop-gradient），ETF 投影头，固定 ETF 原型（C 个已知 + 3 个未知槽），L_BYOL 与 L_src。
- **C 阶段 ② 自适应**：继承参数的编码器，源域 / 目标弱视图 / 目标强视图三路特征，固定 ETF 打分，门控分成三组，L_src、L_ta、L_unk、L_new（待定，紫色虚线）、0.1·L_BYOL；门控分组反过来决定第三章模块的扰动强度。
- **D 推理**：保护阈值判已知 / 未知，未知样本分给最近的未知槽；评估指标、基线、核心消融。
- 图例：蓝色为源域，橙色为目标弱视图，红色为目标强视图 / 未知，绿色为 ETF，灰色虚线为损失 / 监督，紫色虚线为待定的创新损失。
