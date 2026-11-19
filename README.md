# 矿盾声鉴 v2.3 第二轮图库（补充）

只包含第二轮新增的图库，第一轮的图库在 `challenge-cup-data-banks` 分支。两个分支的 `banks/` 合并到同一个文件夹即可。

| 文件后缀 | 内容 |
| --- | --- |
| `_n2` | 白噪声，换了噪声种子（2027），用于确认实验 |
| `_pink` | 粉红噪声 |

`.npz` 放到代码包的 `bench/banks/`，`.mat` 放到 `data/banks/`。由 `bench/make_banks.py` 生成，命令见 `bench/README.md`。
