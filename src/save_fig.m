function save_fig(fig, out_dir, name, cfg)
%SAVE_FIG  保存图片为 PNG
if cfg.save_figures
    print(fig, fullfile(out_dir, [name '.png']), '-dpng', '-r150');
end
end
