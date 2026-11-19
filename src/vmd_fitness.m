function [fit, info] = vmd_fitness(z, x, cfg_vmd)
%VMD_FITNESS  VMD 参数寻优的统一适应度函数（越小越好）
%   z 为归一化参数，内部统一用 vmd_decode 解码。
[alpha, K] = vmd_decode(z, cfg_vmd);
u = vmd_run(x, alpha, K, cfg_vmd);
vals = zeros(K, 1);
for i = 1:K
    switch lower(cfg_vmd.fitness)
        case 'envelope',    vals(i) = envelope_entropy(u(i, :));
        case 'sample',      vals(i) = sample_entropy(u(i, :));
        case 'permutation', vals(i) = permutation_entropy(u(i, :), 3, 1);
        case 'info',        vals(i) = info_entropy(u(i, :));
        otherwise, error('未知的适应度类型: %s', cfg_vmd.fitness);
    end
end
switch lower(cfg_vmd.aggregate)
    case 'min',  fit = min(vals);
    case 'mean', fit = mean(vals);
    otherwise, error('未知的汇总方式: %s', cfg_vmd.aggregate);
end
info = struct('alpha', alpha, 'K', K, 'per_imf', vals);
end
