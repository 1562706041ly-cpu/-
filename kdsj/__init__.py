"""矿盾声鉴（KDSJ）：设备声音/振动信号的分解、时频分析与智能识别软件。

模块：
  audio       读取音频（wav）、分帧
  tfr         同步提取变换时频图、对数梅尔谱、颜色映射
  vmd         变分模态分解、包络熵等适应度指标、VMD 降噪
  optim       优化器：三角拓扑聚合优化（TTAO）、黑翅鸢优化（BKA）、随机搜索，带评估缓存
  nets        识别网络：CNN、ResNet、ECA-ResNet
  lssvm       最小二乘支持向量机（一对一多分类）及 BKA 超参数寻优
  recognizer  识别器：网络特征提取 + BKA-LSSVM 分类，训练、保存、加载、预测
  monitor     对整段音频滑窗识别并生成告警事件
  cli         命令行入口（python -m kdsj）
"""
__version__ = '1.0.0'
SOFTWARE_NAME = '基于Log-Mel与MD的矿山设备异常声音监测系统'   # 软件著作权登记使用的全称
SHORT_NAME = '矿盾声鉴'                                     # 简称
SUBTITLE = '矿山设备异常声音监测系统'                        # 界面副标题
VERSION_LABEL = 'V1.0'
