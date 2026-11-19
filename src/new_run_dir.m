function d = new_run_dir(cfg, tag)
%NEW_RUN_DIR  为每次运行创建带时间戳的结果目录，避免覆盖
d = ensure_dir(fullfile(cfg.result_dir, sprintf('%s_%s', tag, datestr(now, 'yyyymmdd_HHMMSS'))));
end
