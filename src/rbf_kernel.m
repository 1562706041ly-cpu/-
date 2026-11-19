function K = rbf_kernel(A, B, sig2)
%RBF_KERNEL  K(i,j) = exp(-||A_i - B_j||^2 / sig2)（与 LSSVMlab 的 RBF_kernel 定义一致）
D = sum(A.^2, 2) + sum(B.^2, 2).' - 2 * (A * B.');
K = exp(-max(D, 0) / sig2);
end
