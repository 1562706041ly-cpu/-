function [u, omega] = vmd_run(x, alpha, K, cfg_vmd)
%VMD_RUN  按配置调用 VMD
[u, omega] = vmd_fast(x, alpha, cfg_vmd.tau, K, cfg_vmd.DC, cfg_vmd.init, ...
                      cfg_vmd.tol, cfg_vmd.max_inner_iter);
end
