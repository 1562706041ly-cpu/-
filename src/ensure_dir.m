function d = ensure_dir(d)
%ENSURE_DIR  目录不存在时创建
if ~exist(d, 'dir'), mkdir(d); end
end
