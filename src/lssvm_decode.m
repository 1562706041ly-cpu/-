function [gam, sig2] = lssvm_decode(z, cfg_cls)
%LSSVM_DECODE  归一化变量 -> 对数空间的 LSSVM 超参数
z = min(max(z(:).', 0), 1);
g = cfg_cls.log10_gam; s = cfg_cls.log10_sig2;
gam  = 10^(g(1) + z(1) * (g(2) - g(1)));
sig2 = 10^(s(1) + z(2) * (s(2) - s(1)));
end
