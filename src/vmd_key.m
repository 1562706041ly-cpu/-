function k = vmd_key(z, c)
%VMD_KEY  缓存键：解码后相同的 (alpha, K) 视为同一个解
[a, K] = vmd_decode(z, c);
k = sprintf('%d_%d', a, K);
end
