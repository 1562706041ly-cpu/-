# 第三方代码说明

`third_party/` 中的代码**不是本项目原创**，仅作为依赖调用。申请软件著作权时请勿将其计入本项目源代码，并在说明书中注明来源。

| 目录 | 内容 | 来源 / 作者 | 本项目修改 |
|---|---|---|---|
| `TTAO/` | 三角拓扑聚合优化器 | Zhao S., Zhang T., et al. *Expert Systems with Applications*, 2024, DOI: 10.1016/j.eswa.2023.121744 | 改为标准接口 `fobj(x)`；三角单元尺寸按搜索范围缩放；修复 `X00_fit` 被覆盖；去掉迭代打印 |
| `GLO/` | 灰叶猴优化器（Gray Langurs Optimizer） | Barshandeh S., Khodadadi N., et al. *Artificial Intelligence Review*, 2026, DOI: 10.1007/s10462-026-11529-2 | `unifrnd` 替换为等价 `rand` 表达式，去掉统计工具箱依赖 |
| `VMD/` | 变分模态分解原版实现 | Dragomiretskiy K., Zosso D., *IEEE TSP*, 2014 | 未修改，仅作对照；实际运行使用自研 `src/vmd_fast.m` |
| `LSSVMlab/` | LS-SVMlab v1.8 | KU Leuven ESAT-STADIUS，GNU GPL v2 | 未修改；默认不使用（`cfg.cls.lssvm_impl='fast'` 时用自研 `src/lssvm_ovo.m`） |

## 已从原代码包中移除的内容

- 第三方商业教程（“淘个代码”公众号/店铺）的广告、署名与“禁止倒卖”声明；
- `CNN_BKA_LSSVM.m` 中通过 `eval(ax.Tag)` 隐藏执行的语句，解码后为 `disp('作者:<公众号：淘个代码>')`；
- 来源同为该教程的 `BKA.m`、`fun.m`、`initialization.m`、`zjyanseplotConfMat.m`、`axis_control.m`、`Step1.m`、`Step2_JPG_Resize.m`、`data_summary.m`、`FlipLayer.m`，全部由 `src/` 与 `run_*.m` 中的自研代码重写替代；
- 由 MathWorks `wsst` 修改而来的 `wset.m`（MathWorks 版权代码），由自研 `src/set_transform.m` 替代。

如果最终不需要 LSSVMlab 对照，可直接删除 `third_party/LSSVMlab/`，避免 GPL 许可证带来的开源义务。
