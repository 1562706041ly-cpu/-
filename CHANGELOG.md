# 更新记录

## v3.0（2026-11-20）

- 软件主体改为 Python（`kdsj` 包），全部为自写代码，不再依赖 MATLAB、LSSVMlab 和第三方 MATLAB 实现。
- 模块：音频读写、同步提取变换（精确版 + 长音频快速版）、对数梅尔谱、VMD 与降噪、TTAO/BKA 优化、CNN/ResNet/ECA-ResNet、BKA-LSSVM、识别器保存加载、整段音频滑窗告警与 CSV 报告、命令行、14 项自检。
- 新增 MIMII 工业设备声音实验（真实工厂噪声，8 类，两种划分方式），图库已在服务器上生成。
- 齿轮箱振动实验（v2.2/v2.3）的代码和结果在 `challenge-cup-v2.3` 分支。
