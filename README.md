# 矿盾声鉴 v2.2 基准测试图库

由 `bench/make_banks.py` 从东南大学齿轮箱公开数据（第 3 通道）生成。每个文件 1000 张 64×64 时频图（5 类 × 200），保存为 jet 颜色索引（uint8，0–255）。

文件名：`<工况>_<噪声>`，工况为 `20_0`（20 Hz、0 V）或 `30_2`（30 Hz、2 V），噪声为 `clean`、`snr20`、`snr10`、`snr5`、`snr0`、`snr-5`。

| 格式 | 给谁用 | 放到哪里 |
| --- | --- | --- |
| `.npz` | Python（`bench/run_bench.py`） | 代码包的 `bench/banks/` |
| `.mat` | MATLAB（`src/load_tfr_images.m`） | 代码包的 `data/banks/` |

`20_0_clean` 与 v2.1 `run_step2_tfr` 用你们的 `data_total.mat` 生成的 `resizeimg.mat` 逐像素相同。
