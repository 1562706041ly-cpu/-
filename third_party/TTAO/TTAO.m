%__________________________________________________________________
%  Triangulation Topology Aggregation Optimizer
%  Developed in MATLAB R2018a
%
%  programmer: Shijie Zhao and Tianran Zhang   
%  E-mail: zhaoshijie@lntu.edu.cn
%          ztr20010118@126.com
%  The code is based on the following papers.
%  Shijie Zhao, Tianran Zhang, Liang Cai, Ronghua Yang. Triangulation
%  Topology Aggregation Optimizer: A novel mathematics-based meta-heuristic algorithm for engineering applications.
%  Expert Systems with Applications
%  DOI: 10.1016/j.eswa.2023.121744
%
%__________________________________________________________________

function [fbest,Xbest,Convergence_curve]=TTAO(PopSize,T,Low,Up,Dim,fobj,step0)
% ---- 本项目修改说明（相对作者原版）----
% 1) 目标函数改为标准接口 fobj(x)，数据通过匿名函数绑定；
% 2) 三角单元尺寸 l 按搜索范围 (Up-Low) 缩放：l = step0*exp(-t/T).*(Up-Low)，
%    原版固定为 9*exp(-t/T)，当各维量纲差异很大时(如 alpha∈[100,2500]、K∈[3,10])会失效；
% 3) 修复 PopSize 不是 3 的倍数时 X00_fit 被覆盖的问题；去掉迭代打印。
if nargin < 7 || isempty(step0), step0 = 0.15; end
N=floor(PopSize/3); % Initialize N/3 individuals.
X1=rand(N,Dim).*(Up-Low)+Low;
t=1;
while t<T+1
    %% Formation of triangular topological units
    l=step0*exp(-t/T).*(Up-Low);% The size of a triangular topological unit (scaled).
    for i=1:N
        theta=rand(1,Dim)*pi;
        h1=cos(theta);
        h2=cos(theta+pi/3);
        X2(i,:)=X1(i,:)+l.*h1;
        X3(i,:)=X1(i,:)+l.*h2;
    end
    X2 = max(X2,Low);
    X2 = min(X2,Up);
    X3 = max(X3,Low);
    X3 = min(X3,Up);
    r1=rand;r2=rand;
    X4=r1*X1+r2.*X2+(1-r1-r2)*X3;
    X4 = max(X4,Low);
    X4 = min(X4,Up);
    for i=1:N
        X1_fit(i)=fobj(X1(i,:));
        X2_fit(i)=fobj(X2(i,:));
        X3_fit(i)=fobj(X3(i,:));
        X4_fit(i)=fobj(X4(i,:));
    end
    X=[X1 X2 X3 X4];
    fit=[X1_fit;X2_fit;X3_fit;X4_fit];
    [X_sort,index]=sort(fit);
    %% Find the ptimial point and the suboptimial point in each triangular topological unit.
    for i=1:N
        X_best_1(i,:)=X(i,(index(1,i)-1)*Dim+1:index(1,i)*Dim);
        X_best_2(i,:)=X(i,(index(2,i)-1)*Dim+1:index(2,i)*Dim);
    end
    best_fit_1=X_sort(1,:);
    best_fit_2=X_sort(2,:);
    %% Generic aggregation
    for i=1:N
        r=rand(1,Dim);
        X_new=X_best_1;
        X_new(i,:)=[];
        l1=randi(N-1);
        X_G(i,:)=(r.*X_best_1(i,:)+(ones(1,Dim)-r).*X_new(l1,:));
        X_G(i,:) = max(X_G(i,:),Low);
        X_G(i,:) = min(X_G(i,:),Up);
        X_fit_G(i)=fobj(X_G(i,:));
        if X_fit_G(i)<best_fit_1(i)
            X_best_1(i,:)=X_G(i,:);
            best_fit_1(i)=X_fit_G(i);
        elseif X_fit_G(i)<best_fit_2(i)
            X_best_2(i,:)=X_G(i,:);
        end
    end
    %% Local aggregation
    for i=1:N
        a=(exp(1)-(exp(1))^3)/(T-1);
        b=(exp(1))^3-a;
        alpha=log(a*t+b);
        X_C(i,1:Dim)=X_best_1(i,1:Dim)+alpha*(X_best_1(i,1:Dim)-X_best_2(i,1:Dim));
        X_C(i,:) = max(X_C(i,:),Low);
        X_C(i,:) = min(X_C(i,:),Up);
        X_fit_C(i)=fobj(X_C(i,:));
        if X_fit_C(i)<best_fit_1(i)
            X_best_1(i,:)=X_C(i,:);
            best_fit_1(i)=X_fit_C(i);
        end
    end
    %% 
    N00=PopSize-N*3;
    if N00~=0
        X00=rand(PopSize-N*3,Dim).*(Up-Low)+Low;
        X00_fit=zeros(1,N00);
        for i=1:N00
            X00_fit(i)=fobj(X00(i,:));
        end
        X_1_0=[X_best_1;X00];
        X_1_0_fit=[best_fit_1,X00_fit];
        [~,index01]=sort(X_1_0_fit);
        X_best_1=X_1_0(index01(1:N),:);
        best_fit_1=X_1_0_fit(index01(1:N));
    end
    X1=X_best_1;
    %% The first N/3 individuals serve as the initial population for the next iteration.
    [~,index1]=min(best_fit_1);
    Xbest=X1(index1,:);
    fbest=best_fit_1(index1);
    Convergence_curve(t)=fbest;
    t = t + 1;
end

