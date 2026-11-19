function z = vmd_encode(alpha, K, cfg_vmd)
%VMD_ENCODE  vmd_decode 的逆变换（用于网格搜索结果、初始解等）
ar = cfg_vmd.alpha_range; kr = cfg_vmd.K_range;
nK = kr(2) - kr(1) + 1;
z = [(alpha - ar(1)) / (ar(2) - ar(1)), (K - kr(1) + 0.5) / nK];
end
