function res = run_optimizer(name, counter, dim, pop, max_iter, opts)
%RUN_OPTIMIZER  统一调用各优化器（均在归一化空间 [0,1]^dim 中搜索）
%   counter 为 EvalCounter；预算耗尽时自动停止并返回当前最优。
if nargin < 6, opts = struct(); end
lb = zeros(1, dim); ub = ones(1, dim);
fobj = @(z) counter.eval(z);
t0 = tic;
try
    switch upper(name)
        case 'TTAO'
            step = 0.15; if isfield(opts, 'ttao_step'), step = opts.ttao_step; end
            TTAO(pop, max_iter, lb, ub, dim, fobj, step);
        case 'GLO'
            GLO(pop, max_iter, lb, ub, dim, fobj);
        case 'BKA'
            bka_optimize(pop, max_iter, lb, ub, dim, fobj);
        case 'RANDOM'
            for k = 1:pop * max_iter
                fobj(rand(1, dim));
            end
        otherwise
            error('未知优化器: %s', name);
    end
catch err
    if ~strcmp(err.identifier, 'EvalCounter:budget'), rethrow(err); end
end
res.name     = upper(name);
res.best_fit = counter.best_fit;
res.best_z   = counter.best_z;
res.info     = counter.best_info;
res.trace    = counter.trace;
res.n_calls  = counter.n_calls;
res.n_unique = counter.n_unique;
res.time_s   = toc(t0);
end
