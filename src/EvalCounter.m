classdef EvalCounter < handle
%EVALCOUNTER  适应度调用计数 + 结果缓存 + 预算控制
%   - 相同参数（解码后相同的 alpha、K）不重复做 VMD，节省大量时间；
%   - 记录"评估次数-当前最优"曲线，用于公平比较不同优化器；
%   - 超出预算时抛出 'EvalCounter:budget' 异常，由调用方捕获。
    properties
        fun                 % @(z) -> [fit, info]
        keyfun              % @(z) -> 缓存键
        max_evals = Inf
        n_calls = 0
        n_unique = 0
        best_fit = Inf
        best_z = []
        best_info = []
        trace = zeros(0, 1) % 每次调用后的最优适应度
        cache
    end
    methods
        function obj = EvalCounter(fun, keyfun, max_evals)
            obj.fun = fun;
            obj.keyfun = keyfun;
            if nargin >= 3 && ~isempty(max_evals), obj.max_evals = max_evals; end
            obj.cache = containers.Map('KeyType', 'char', 'ValueType', 'any');
        end
        function f = eval(obj, z)
            if obj.n_calls >= obj.max_evals
                error('EvalCounter:budget', '适应度评估预算已用完');
            end
            key = obj.keyfun(z);
            if isKey(obj.cache, key)
                r = obj.cache(key);
            else
                [r.fit, r.info] = obj.fun(z);
                obj.cache(key) = r;
                obj.n_unique = obj.n_unique + 1;
            end
            f = r.fit;
            obj.n_calls = obj.n_calls + 1;
            if f < obj.best_fit
                obj.best_fit = f; obj.best_z = z(:).'; obj.best_info = r.info;
            end
            obj.trace(end+1, 1) = obj.best_fit;
        end
    end
end
