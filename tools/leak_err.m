function e = leak_err(z, Fd, yd, Ft, yt, c)
%LEAK_ERR  仅用于演示原代码的错误做法：直接用测试集错误率当适应度（请勿在正式流程中使用）
[g, s2] = lssvm_decode(z, c);
e = mean(lssvm_ovo(Fd, yd, Ft, g, s2, 5) ~= yt);
end
