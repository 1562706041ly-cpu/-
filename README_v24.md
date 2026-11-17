# v24 (pump, GPU): 纯最近邻 K 与距离能量 K 对 Rg 的端到端对照

只用 Bruinsma 泵数据(Motor-2,100% 转速)。75% 转速封存数据不打开。

## 放哪里
把本分支的 4 个文件放到你放 `m2_paderborn_kr.py` 的同一个文件夹(`D:\DCWRU_Data\Paderborn_Experiment\`):
`run_batch_v24_cf.ps1`、`analyze_v24.py`、`v24_endtoend_PREREGISTERED.txt`、本说明。
`runs_paderborn\V20_*_Rg_seed*` 必须还在(v23 的 27 个 Rg 结果,含 results_exp2.json 和 pred 文件)。不要删除。

## 先查再跑
1. 脚本一开始会打印 `m2_paderborn_kr.py` 和 `kr_modules.py` 的 sha256,必须等于 v23 版本,否则自动停止并提示。
   期望值:m2_paderborn_kr.py 3f5ffb6a...6443827,kr_modules.py 7dfe0e59...a303f(完整值写在脚本里)。
2. 如果停止,说明本地代码不是 v23 版本。此时不要运行,告诉我,我会安排三组全部重跑。

## 运行(约 1 到 1.5 小时,是按 v22 批次推的估计,没有实测)
    cd D:\DCWRU_Data\Paderborn_Experiment
    .\run_batch_v24_cf.ps1
新增 57 个运行:Rk 27 个,Rd 27 个,Rgrep 3 个(重复 Rg,用来量 GPU 的运行间噪声)。已存在的会自动跳过,中断后可直接重跑。

## 结果
运行结束自动调用 `analyze_v24.py`,先做配对核验(参数、标注窗口索引、样本数逐个对比),不通过就中止。
需要上传到 GitHub(新分支或你自己的分支):`summary_v24.txt`、`summary_v24.csv`、`pairing_report_v24.txt`。
也可以单独运行:  `python analyze_v24.py --root runs_paderborn`
