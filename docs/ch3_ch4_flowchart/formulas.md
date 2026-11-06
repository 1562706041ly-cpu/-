# 第三、四章公式汇总（草稿，待核对原稿；LaTeX 写法）


## 图 1  第三、四章技术路线图

## 0  符号说明

## 第三章  已知性与可靠度联合控制的自适应增强

### 3.1  输入
> 【待核对】第三章原稿尚未提供，输入的具体设定以原稿为准。

### 3.2  已知性 K(x)：能量 + 原型距离
> 【部分有依据】前期实验用过“与已知类原型的最大相似度”作为已知性分数（对应下面的 s(x)）；能量项以及两项加权的形式、权重均为推测，不代表论文最终公式。
$$E(x)=-T\log\sum_{c=1}^{C}\exp\left(l_c(x)/T\right)\qquad(3-1)$$
$$e(x)=\sigma\left(\frac{-E(x)-\tau_E}{\gamma}\right)\qquad(3-2)$$
$$s(x)=\exp\left(-\frac{d_{\min}^{2}(x)}{2\sigma_d^{2}}\right),\quad d_{\min}(x)=\min_{c}\|z-\mu_c\|_2\qquad(3-3)$$
$$K(x)=\lambda\,e(x)+(1-\lambda)\,s(x),\quad K(x)\in[0,1]\qquad(3-4)$$

### 3.3  可靠度 R(x)：三项一致性
> 【推测】前期仅观察到低可靠度的已知样本更容易分错；R 由哪几项构成、如何合并（此处用几何平均）均待核对。其中 R_s 里令 p(x)=K(x) 纯属假设，“分流”的含义和分流概率的来源尚未确定，请勿当作确定结构。
$$R_c(x)=1-\left|e(x)-s(x)\right|\qquad(3-5)$$
$$R_v(x)=\frac{1}{2}\left(1+\frac{z^{(1)\top}z^{(2)}}{\|z^{(1)}\|\,\|z^{(2)}\|}\right)\qquad(3-6)$$
$$R_s(x)=\left|2p(x)-1\right|,\quad p(x)=K(x)\qquad(3-7)$$
$$R(x)=\left(R_c(x)\,R_v(x)\,R_s(x)\right)^{1/3}\qquad(3-8)$$

### 3.4  样本证据分区（软权重，不做硬划分）
> 【推测】“可靠已知/可靠未知/不确定”的软权重写法是按流程图推的，是否这样定义待核对。
$$w_{k}(x)=K(x)R(x),\quad w_{u}(x)=\left(1-K(x)\right)R(x),\quad w_{n}(x)=1-R(x)\qquad(3-9)$$
$$w_{k}(x)+w_{u}(x)+w_{n}(x)=1\qquad(3-10)$$

### 3.5  增强预算分配 b(x) = f(K, R)
> 【推测】研究方向（用 K、R 控制增强程度）与设想一致，但 b(x) 的函数形式、扰动强度和模态数的映射均为推测。
$$b(x)=R(x)^{\gamma_b}\left[\kappa_u+(\kappa_k-\kappa_u)K(x)\right]\qquad(3-11)$$
$$\varepsilon(x)=\varepsilon_{\max}\,b(x),\qquad m(x)=\max\left(1,\lceil M\,b(x)\rceil\right)\qquad(3-12)$$

### 3.6  VMD 分解：得到各本征模态分量
> 【部分有依据】VMD 本身是标准方法。本文档用 M 表示模态数，以免与已知性 K(x) 混淆（前期实验里的 K=8 指模态数）。前期 VMD 为信号级可行性实验，不等于已完成 VMD 增强模型训练。
$$\min_{\{u_k\},\{\omega_k\}}\ \sum_{k=1}^{M}\left\|\partial_t\left[\left(\delta(t)+\frac{j}{\pi t}\right)*u_k(t)\right]e^{-j\omega_k t}\right\|_2^{2}\qquad(3-13)$$
$$\mathrm{s.t.}\quad \sum_{k=1}^{M}u_k(t)=s(t)\qquad(3-14)$$

### 3.7  按预算扰动模态幅值，并重构增强视图
> 【推测】按能量选模态、均匀随机缩放的具体做法是示意，待核对。
$$\mathcal{S}(x)=\mathrm{TopK}_{k}\left(\|u_k\|_2^{2},\ m(x)\right)\qquad(3-15)$$
$$\tilde u_k=(1+\epsilon_k)u_k\ (k\in\mathcal{S}),\quad \tilde u_k=u_k\ (k\notin\mathcal{S}),\quad \epsilon_k\sim U\left(-\varepsilon(x),\varepsilon(x)\right)\qquad(3-16)$$
$$\tilde s(t)=\sum_{k=1}^{M}\tilde u_k(t)\qquad(3-17)$$

### 3.8  保真检查：相对扰动量 / 频谱结构 / 整体振动能量
> 【部分有依据】前期 VMD 实验报告过相对扰动量、RMS 相对变化和频谱形状相似度，并发现模态扰动会改变频谱形状，所以需要保真约束。JS 散度写法和全部阈值均未确认，请勿把实验观察值当阈值。
$$\delta_{\mathrm{rel}}=\frac{\|\tilde s-s\|_2}{\|s\|_2}\qquad(3-18)$$
$$D_{\mathrm{spec}}=\mathrm{JS}\left(P_s\,\|\,P_{\tilde s}\right),\quad P_s(f)=\frac{|\mathcal{F}[s](f)|^{2}}{\sum_f|\mathcal{F}[s](f)|^{2}}\qquad(3-19)$$
$$\rho_E=\left|\frac{\|\tilde s\|_2^{2}}{\|s\|_2^{2}}-1\right|\qquad(3-20)$$
$$\mathrm{pass}\Longleftrightarrow \delta_{\mathrm{rel}}\leq\delta_{\max}\ \wedge\ D_{\mathrm{spec}}\leq\tau_s\ \wedge\ \rho_E\leq\tau_e\qquad(3-21)$$
$$\varepsilon(x)\leftarrow\eta\,\varepsilon(x)\ (0<\eta<1),\quad \text{重试 }T_{\max}\text{ 次；仍失败则回退 }b(x)=b_{\min}\qquad(3-22)$$

## 第四章  分流引导的可靠正负关系构建与对比学习

### 4.1  输入
> 【待核对】第四章输入与符号以原稿为准。
$$s_{ij}=\frac{z_i^{\top}z_j}{\|z_i\|\,\|z_j\|}\qquad(4-1)$$

### 4.2  三类关系证据
> 【部分有依据】结合标签锚点、局部邻域、跨视图信息判断关系，这个动机有依据；三项证据的具体公式（Jaccard 等）为推测。
$$a_{ij}=+1\ (y_i=y_j);\quad -1\ (y_i\neq y_j);\quad 0\ (\mathrm{unlabeled})\qquad(4-2)$$
$$J_{ij}=\frac{|\mathcal{N}_k(i)\cap\mathcal{N}_k(j)|}{|\mathcal{N}_k(i)\cup\mathcal{N}_k(j)|}\qquad(4-3)$$
$$n_{ij}=+J_{ij}\ \ (j\in\mathcal{N}_k(i)\ \mathrm{and}\ i\in\mathcal{N}_k(j))\qquad(4-4)$$
$$n_{ij}=-(1-J_{ij})\ \ (j\notin\mathcal{N}_k(i),\ i\notin\mathcal{N}_k(j),\ J_{ij}<\tau_J);\quad n_{ij}=0\ \ (\mathrm{otherwise})$$
$$A_{ij}=\mathbf{1}\left[(s^{(1)}_{ij}-\tau_s)(s^{(2)}_{ij}-\tau_s)>0\right],\quad c_{ij}=A_{ij}\left(1-\left|s^{(1)}_{ij}-s^{(2)}_{ij}\right|\right)\qquad(4-5)$$

### 4.3  关系可信度评估（正 / 负关系是否可信）
> 【部分有依据】动机有依据：前期诊断发现相似度很高的“负样本”可能实际同类，仅按相似度剔除负样本也不稳定。融合公式、阈值与正负对选择规则为推测。使用真实标签的 O1/O2 只是上限分析，不是可直接使用的方法。
$$t_{ij}=\alpha_1 a_{ij}+\alpha_2 n_{ij}\qquad(4-6)$$
$$q_{ij}=c_{ij}\cdot\sqrt{R_iR_j}\cdot\min\left(1,|t_{ij}|\right)\qquad(4-7)$$
$$\mathcal{P}=\{(i,j):t_{ij}>0,\ q_{ij}\geq\tau_q\},\quad \mathcal{N}=\{(i,j):t_{ij}<0,\ q_{ij}\geq\tau_q\},\quad \mathcal{U}=\text{rest}\qquad(4-8)$$
$$w_{ij}=q_{ij}\ \ ((i,j)\in\mathcal{P}\cup\mathcal{N});\quad w_{ij}=\lambda_u q_{ij}\ \ ((i,j)\in\mathcal{U},\ \lambda_u\ \mathrm{small\ or}\ 0)\qquad(4-9)$$

### 4.4  选择性难负对学习
> 【推测】难负对的挖掘和假负剔除规则为推测。
$$\mathcal{H}_i=\mathrm{TopH}_{k\in\mathcal{N}_i}\left(s_{ik}\right)\ \ \mathrm{s.t.}\ \ s_{ik}\leq\tau_{fn}\qquad(4-10)$$

### 4.5  联合优化
> 【待核对】损失函数是否恰为这三项、权重如何，以原稿为准。
$$\mathcal{L}_{ucl}=-\frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}}w_{ij}\log\frac{\exp(s_{ij}/\tau)}{\exp(s_{ij}/\tau)+\sum_{k\in\mathcal{H}_i}w_{ik}\exp(s_{ik}/\tau)}\qquad(4-11)$$
$$\mathcal{L}_{sup}=-\frac{1}{|B_l|}\sum_{i\in B_l}\frac{1}{|P_l(i)|}\sum_{p\in P_l(i)}\log\frac{\exp(s_{ip}/\tau)}{\sum_{a\ne i}\exp(s_{ia}/\tau)}\qquad(4-12)$$
$$\mathcal{L}_{cls}=-\frac{1}{|B_l|}\sum_{i\in B_l}\log p(y_i\mid x_i)\qquad(4-13)$$
$$\mathcal{L}=\mathcal{L}_{sup}+\lambda_1\,\mathcal{L}_{ucl}+\lambda_2\,\mathcal{L}_{cls}\qquad(4-14)$$

### 4.6  输出：已知类识别与未知类聚类发现
> 【部分有依据】前期基线用过 GCD 特征学习加半监督 K-means；也探索过冻结编码器后设两个未知聚类中心（均映射为“未知”），但那只是探索性对照，不代表最终类别数。用轮廓系数选聚类数为推测。
$$\hat y(x)=\mathrm{arg\,max}_{c}\ p(c\mid x),\quad x\in\mathcal{X}_{k}=\{x:K(x)\geq\tau_K\}\qquad(4-15)$$
$$\mathcal{X}_{u}=\{x:K(x)<\tau_K,\ R(x)\geq\tau_R\}\qquad(4-16)$$
$$\hat k=\mathrm{arg\,max}_{k}\ \mathrm{Sil}(k),\quad \mathrm{Sil}(k)=\frac{1}{n}\sum_{i=1}^{n}\frac{b_i-a_i}{\max(a_i,b_i)}\qquad(4-17)$$

### 4.7  迭代：更新编码器 → 重新评估 K、R
> 【待核对】迭代方式以原稿为准。
$$\theta^{(t+1)}=\theta^{(t)}-\eta_{lr}\nabla_\theta\mathcal{L},\quad \{\mu_c,K,R\}^{(t+1)}\leftarrow f_{\theta^{(t+1)}}\qquad(4-18)$$

## 附：整体流程（算法概要）