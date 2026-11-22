# 依赖、算法来源与数据集

## 源代码

`kdsj/` 中的代码全部由本项目编写，没有复制第三方源代码。原 MATLAB 教程代码（"淘个代码"）、MathWorks 的 `wset.m`、TTAO/GLO/VMD 作者的 MATLAB 实现、LSSVMlab 都没有进入 Python 版。

## 运行依赖（只调用，不修改）

| 库 | 用途 | 许可证 |
| --- | --- | --- |
| NumPy | 数值计算、FFT | BSD-3-Clause |
| PyTorch | 神经网络训练与推理 | BSD-3-Clause |
| PySide6（Qt for Python） | 图形界面 | LGPL-3.0（以动态链接方式使用，未修改；可以用于闭源软件，发布时需注明并允许用户替换该库） |

申请软著时，这些库作为运行环境写在说明书里，不计入源代码。

## 按论文自行实现的算法

| 算法 | 文献 |
| --- | --- |
| 变分模态分解 VMD | Dragomiretskiy K., Zosso D. Variational mode decomposition. *IEEE Trans. Signal Processing*, 2014, 62(3): 531–544 |
| 同步提取变换 SET | Yu G., Yu M., Xu C. Synchroextracting transform. *IEEE Trans. Industrial Electronics*, 2017, 64(10): 8042–8054 |
| 三角拓扑聚合优化 TTAO | Zhao S., Zhang T., Cai L., Yang R. Triangulation topology aggregation optimizer. *Expert Systems with Applications*, 2024, 238: 121744 |
| 黑翅鸢优化 BKA | Wang J., Wang W., Hu X., et al. Black-winged kite algorithm. *Artificial Intelligence Review*, 2024, 57: 98 |
| LS-SVM | Suykens J.A.K., Vandewalle J. Least squares support vector machine classifiers. *Neural Processing Letters*, 1999, 9: 293–300 |
| 残差网络 | He K., Zhang X., Ren S., Sun J. Deep residual learning for image recognition. *CVPR*, 2016 |
| ECA 注意力 | Wang Q., Wu B., Zhu P., et al. ECA-Net. *CVPR*, 2020 |

## 界面素材

- 背景照片（`kdsj/assets/background.jpg`）由使用者自行提供，需确认版权。代码包里没有附带图片。
- 界面截图（`docs/`）是在服务器上渲染的，使用 Noto Sans SC 字体（SIL OFL 1.1），该字体没有随软件分发；Windows 上界面使用系统自带的微软雅黑。

## 数据集

| 数据 | 用途 | 许可 |
| --- | --- | --- |
| MIMII（Hitachi, Ltd.），<https://zenodo.org/records/3384388> | 工业设备声音识别实验 | CC BY-SA 4.0。使用时须注明来源；由它生成的图库再公开时须同样采用 CC BY-SA 4.0 |
| 东南大学齿轮箱数据，<https://github.com/cathysiyu/Mechanical-datasets> | 振动信号实验（v2.x） | 按原作者要求引用：Shao S. et al., *IEEE Trans. Industrial Informatics*, 2019, 15(4): 2446–2455 |

MIMII 引用：Purohit H., Tanabe R., Ichige K., et al. MIMII Dataset: Sound dataset for malfunctioning industrial machine investigation and inspection. *Proc. DCASE Workshop*, 2019.
