# 更新记录

## V1.0（对外发布版本，2026-11-22）

- 软件全称定为“基于Log-Mel与MD的矿山设备异常声音监测系统”，简称“矿盾声鉴”，版本号 V1.0（`kdsj/__init__.py` 中的 SOFTWARE_NAME、SHORT_NAME、VERSION_LABEL）。下面的 v3.x 是开发过程中的内部版本号。
- 合并 v3.2.1、v3.2.2 的界面修改；报告中的统计项改为中文；波形区未分析时的提示改为“尚未分析录音”。
- 新增软著材料生成脚本：`tools/make_code_doc.py`（源程序文档）、`tools/make_screenshots.py`（界面截图）、`tools/make_manual.py` + `tools/manual_content.py`（用户手册）。生成结果在 `软著材料/`。

## v3.2（2026-11-20）

- 新增图形界面 `kdsj/gui/`（PySide6）：登录页（账号任意，密码 123456，只保存哈希）、监测主页（设备列表、录音波形、时频图、异常指数曲线、运行状态卡片、告警记录、导出报告）、新建设备对话框（只用正常录音建模）。建模和分析在后台线程运行，界面不会卡。
- 新增 `启动矿盾声鉴.bat`、`tools/gui_preview.py`（无窗口渲染界面截图）。
- `anomaly.train_from_files` 供命令行和界面共用。
- 8 类识别补到 10 个种子：BKA-LSSVM 在 40 次运行中全部提升（p = 0.002）；梅尔谱优于同步提取变换图（p = 0.004）；梅尔谱下 ResNet 与 CNN 无显著差别。

## v3.1（2026-11-20）

- 新增 `kdsj/anomaly.py`：只用正常声音训练的异常检测（频带统计 + 马氏距离、自编码器、设备编号分类），可保存加载。
- 新增命令 `python -m kdsj train-anomaly`；`analyze` 自动识别分类模型和异常检测模型。
- 新增 `experiments/mimii_anomaly.py`、`mimii_anomaly_summary.py`，以及 MIMII 两类实验的结果和结论（`results/README.md`）。
- 自检增加到 15 项。

## v3.0（2026-11-20）

- 软件主体改为 Python（`kdsj` 包），全部为自写代码，不再依赖 MATLAB、LSSVMlab 和第三方 MATLAB 实现。
- 模块：音频读写、同步提取变换（精确版 + 长音频快速版）、对数梅尔谱、VMD 与降噪、TTAO/BKA 优化、CNN/ResNet/ECA-ResNet、BKA-LSSVM、识别器保存加载、整段音频滑窗告警与 CSV 报告、命令行、14 项自检。
- 新增 MIMII 工业设备声音实验（真实工厂噪声，8 类，两种划分方式），图库已在服务器上生成。
- 齿轮箱振动实验（v2.2/v2.3）的代码和结果在 `challenge-cup-v2.3` 分支。
