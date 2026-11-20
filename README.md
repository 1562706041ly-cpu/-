# MIMII 时频图库（矿盾声鉴 v3.0 实验用）

把 `mimii_banks/` 文件夹放到代码包根目录（与 `kdsj/` 同级）。

每个文件对应一种设备（fan 风机、pump 水泵、slider 滑轨、valve 阀门），内容：
- `X_set`：同步提取变换图，N x 64 x 64 uint8 颜色索引
- `X_mel`：对数梅尔谱图，同上
- `id`：设备编号（0/2/4/6）；`state`：0 正常、1 异常；`name`：原始文件名

## 来源与许可

由 MIMII 数据集 0 dB 子集第 1 通道生成（`experiments/mimii_extract.py`、`experiments/mimii_banks.py`）。

- 原始数据：Hitachi, Ltd., MIMII Dataset, <https://zenodo.org/records/3384388>
- 引用：Purohit H., Tanabe R., Ichige K., et al. MIMII Dataset: Sound dataset for malfunctioning industrial machine investigation and inspection. Proc. DCASE Workshop, 2019.
- 许可：CC BY-SA 4.0。本图库是 MIMII 的衍生数据，同样以 CC BY-SA 4.0 提供。
