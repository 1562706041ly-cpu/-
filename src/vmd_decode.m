function [alpha, K] = vmd_decode(z, cfg_vmd)
%VMD_DECODE  把优化器的归一化变量 z∈[0,1]^2 解码为 VMD 参数
%   适应度计算和最终分解都只调用这个函数，保证两处参数完全一致
%   （修复原代码中适应度用 fix、最终分解用 round 导致 K 不一致的问题）。
z = min(max(z(:).', 0), 1);
ar = cfg_vmd.alpha_range; kr = cfg_vmd.K_range;
alpha = round(ar(1) + z(1) * (ar(2) - ar(1)));
nK = kr(2) - kr(1) + 1;                       % 每个整数 K 占相同宽度
K = kr(1) + min(floor(z(2) * nK), nK - 1);
end
