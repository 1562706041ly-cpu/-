function [f, info] = cached_fitness(z, x, c, shared)
%CACHED_FITNESS  多次运行共享的 VMD 适应度缓存（shared 为 containers.Map）
k = vmd_key(z, c);
if isKey(shared, k)
    r = shared(k); f = r.f; info = r.info;
else
    [f, info] = vmd_fitness(z, x, c);
    shared(k) = struct('f', f, 'info', info);
end
end
