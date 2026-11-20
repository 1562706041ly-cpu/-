"""优化器（都在归一化空间 [0,1]^dim 中最小化目标函数）。

- ttao   三角拓扑聚合优化器（Zhao S. et al., Expert Systems with Applications, 2024,
         DOI 10.1016/j.eswa.2023.121744）。本项目按论文自行实现，三角单元尺寸按搜索范围比例设置。
- bka    黑翅鸢优化算法（Wang J. et al., Artificial Intelligence Review, 2024）。本项目按论文自行实现。
- random 随机搜索，作为对照。

所有优化器都通过 Evaluator 调用目标函数：相同参数只算一次（缓存），记录收敛曲线，可设评估预算。
"""
import time
import numpy as np


class Budget(Exception):
    pass


class Evaluator:
    """目标函数包装：缓存 + 计数 + 预算 + 收敛曲线。

    fun(z) -> (适应度, 附加信息)；keyfun(z) -> 可哈希的缓存键（例如解码后的整数参数）。
    """

    def __init__(self, fun, keyfun=None, max_evals=None):
        self.fun, self.keyfun, self.max_evals = fun, keyfun or (lambda z: tuple(np.round(z, 6))), max_evals
        self.cache, self.n_calls = {}, 0
        self.best_fit, self.best_z, self.best_info, self.trace = np.inf, None, None, []

    @property
    def n_unique(self):
        return len(self.cache)

    def __call__(self, z):
        if self.max_evals is not None and self.n_calls >= self.max_evals:
            raise Budget()
        z = np.clip(np.asarray(z, float), 0, 1)
        k = self.keyfun(z)
        if k not in self.cache:
            self.cache[k] = self.fun(z)
        fit, info = self.cache[k]
        self.n_calls += 1
        if fit < self.best_fit:
            self.best_fit, self.best_z, self.best_info = fit, z.copy(), info
        self.trace.append(self.best_fit)
        return fit


def ttao(f, dim, pop, iters, rng, step0=0.15):
    """三角拓扑聚合优化。pop 个个体中 N = pop // 3 个作为三角单元的顶点，其余随机补充。"""
    N = max(pop // 3, 2)
    extra = pop - 3 * N if pop >= 6 else 0
    P = rng.random((N, dim))
    a = (np.e - np.e ** 3) / max(iters - 1, 1)
    b = np.e ** 3 - a
    for t in range(1, iters + 1):
        l = step0 * np.exp(-t / iters)                         # 三角单元尺寸（相对搜索范围）
        th = rng.random((N, dim)) * np.pi
        V2 = np.clip(P + l * np.cos(th), 0, 1)                 # 以 P 为顶点构造正三角形的另两个顶点
        V3 = np.clip(P + l * np.cos(th + np.pi / 3), 0, 1)
        r1, r2 = rng.random(), rng.random()
        V4 = np.clip(r1 * P + r2 * V2 + (1 - r1 - r2) * V3, 0, 1)  # 单元内部的一个点
        cand = np.stack([P, V2, V3, V4], 1)                    # N x 4 x dim
        fit = np.array([[f(c) for c in unit] for unit in cand])
        order = np.argsort(fit, 1)
        best = cand[np.arange(N), order[:, 0]].copy()
        second = cand[np.arange(N), order[:, 1]].copy()
        fb = fit[np.arange(N), order[:, 0]].copy()
        fs = fit[np.arange(N), order[:, 1]].copy()
        # 一般聚合：每个单元的最优点与另一个随机单元的最优点做随机线性组合
        for i in range(N):
            j = rng.choice([k for k in range(N) if k != i])
            r = rng.random(dim)
            g = np.clip(r * best[i] + (1 - r) * best[j], 0, 1)
            fg = f(g)
            if fg < fb[i]:
                best[i], fb[i] = g, fg
            elif fg < fs[i]:
                second[i], fs[i] = g, fg
        # 局部聚合：沿"次优 -> 最优"方向外推，步长随迭代从 3 递减到 1
        lam = np.log(a * t + b)
        for i in range(N):
            c = np.clip(best[i] + lam * (best[i] - second[i]), 0, 1)
            fc = f(c)
            if fc < fb[i]:
                best[i], fb[i] = c, fc
        # 随机补充个体参与竞争，保留最好的 N 个作为下一代顶点
        if extra:
            R = rng.random((extra, dim))
            fr = np.array([f(x) for x in R])
            allp, allf = np.r_[best, R], np.r_[fb, fr]
            keep = np.argsort(allf)[:N]
            best, fb = allp[keep], allf[keep]
        P = best


def bka(f, dim, pop, iters, rng, p=0.9):
    """黑翅鸢优化。修正了网上流传版本的三处错误：随机数 r 每个个体每次重新生成；
    柯西变异每一维独立采样；每代最优从整个种群正确选取。"""
    X = rng.random((pop, dim))
    F = np.array([f(x) for x in X])
    ib = int(F.argmin())
    gbest, gfit = X[ib].copy(), F[ib]
    for t in range(1, iters + 1):
        n = 0.05 * np.exp(-2 * (t / iters) ** 2)
        for i in range(pop):
            r = rng.random()
            # 攻击行为
            if p < r:
                Xn = X[i] + n * (1 + np.sin(r)) * X[i]
            else:
                Xn = X[i] * (n * (2 * rng.random(dim) - 1) + 1)
            Xn = np.clip(Xn, 0, 1)
            fn = f(Xn)
            if fn < F[i]:
                X[i], F[i] = Xn, fn
            # 迁徙行为（柯西变异）
            m = 2 * np.sin(r + np.pi / 2)
            s = rng.integers(pop)
            c = np.tan((rng.random(dim) - 0.5) * np.pi)
            Xn = X[i] + c * (X[i] - gbest) if F[i] < F[s] else X[i] + c * (gbest - m * X[i])
            Xn = np.clip(Xn, 0, 1)
            fn = f(Xn)
            if fn < F[i]:
                X[i], F[i] = Xn, fn
            if F[i] < gfit:
                gbest, gfit = X[i].copy(), F[i]


def random_search(f, dim, pop, iters, rng):
    for _ in range(pop * iters):
        f(rng.random(dim))


OPTIMIZERS = {'ttao': ttao, 'bka': bka, 'random': random_search}


def optimize(name, evaluator, dim, pop=12, iters=20, seed=0, **kw):
    """运行优化器，预算用完时自动停止。返回结果字典。"""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    try:
        OPTIMIZERS[name.lower()](evaluator, dim, pop, iters, rng, **kw)
    except Budget:
        pass
    return {'name': name.upper(), 'best_fit': evaluator.best_fit, 'best_z': evaluator.best_z,
            'info': evaluator.best_info, 'trace': np.array(evaluator.trace), 'n_calls': evaluator.n_calls,
            'n_unique': evaluator.n_unique, 'time_s': time.time() - t0}
