function write_table_csv(file, header, rows)
%WRITE_TABLE_CSV  写 CSV（UTF-8 BOM，Excel 打开中文不乱码）；rows 为元胞数组
fid = fopen(file, 'w', 'n', 'UTF-8');
fwrite(fid, uint8([239 187 191]));
fprintf(fid, '%s\n', strjoin(header, ','));
for i = 1:size(rows, 1)
    c = rows(i, :);
    for j = 1:numel(c)
        if isnumeric(c{j}), c{j} = num2str(c{j}, '%.6g'); end
    end
    fprintf(fid, '%s\n', strjoin(c, ','));
end
fclose(fid);
end
