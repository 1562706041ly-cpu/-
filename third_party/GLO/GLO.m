function [Best_Cost, Best_Pos, Convergence_curve] = GLO(N, mt, lb, ub, D, fobj)
% 本项目修改：将统计工具箱函数 unifrnd 替换为等价的 rand 表达式，去掉工具箱依赖。
%__________________________________________________________________ 
% %                                                                   % 
% %                                                                   % 
% %                    Gray Langurs Optimizer (GLO)                   % 
% %                                                                   % 
% %                                                                   % 
% %               Developed in MATLAB R2024b (MacOs-Monterey)         % 
% %                                                                   % 
% %                                                                   % 
% %                                                                   % 
% %                                                                   % 
% %                                                                   % 
% %                            e-Mail                                 % 
% %                ---------------------------------                  % 
% %                         nimakhan@berkeley.edu                     % 
%                                                                     % 
% %                                                                   % 
% %                                                                   %  
% %                    https://nimakhodadadi.com                      % 
% %                                                                   % 
% %                                                                   % 
% %                                                                   % 
% %                                                                   % 
% %                        Cite this article                          % 
% %       Barshandeh, S., Khodadadi, N., Abdollahzadeh, B. et al.     % 
% %  Gray langurs optimizer: a multi-group bio-inspired optimization  % 
% %               algorithm. Artif Intell Rev (2026)                  % 
% %           https://doi.org/10.1007/s10462-026-11529-2              % 
% %                                                                   % 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    %% Initialization (Moved from Run.m)
    if numel(lb) == 1
        lb = repmat(lb, 1, D);
        ub = repmat(ub, 1, D);
    end
    
    for i = 1 : N
        Sol(i).X = lb + rand(size(lb)).*(ub - lb);   %#ok
        Sol(i).Cost = fobj(Sol(i).X);   %#ok
    end
    [~,index] = min([Sol.Cost]);
    Best = Sol(index);

    %% Setting parameters
    FP_om = 0.4;   % Female percentage in one-male group
    FP_mm = 0.3;   % Female percentage in multi-male group
    MP_mm = 0.3;   % Male   percentage in multi-male group
    omega = 0.05;  % Migration percentage
    phi = 5;       % Minimum member count
    
    %% Initialization
    Group = Initialization(Sol, N);
    Convergence_curve = zeros(1, mt);
    
    %% Optimization
    for ct = 1 : mt
        ASF = 0.1 + (1 - ct/mt)^2 * .9;   % Adaptive Scaling factor
        
        %% Group 1: One-male (including women and children)
        S_om = numel(Group(1).Member); % Size of all male group
        Nf = floor(S_om*FP_om);        % Eq. (11) number of female langurs
        
        for i = 1 : S_om
            if i == 1
                % The first langur is alpha
                new.X = update_alpha(Group(1).Member(i).X, Best.X, ASF, ct, mt, lb, ub, D);
            elseif i <= Nf
                % The next Nf langures are females
                new.X = update_female(Group(1).Member, 1, i, D, ASF); 
            else
                % The ramaining langures are children
                new.X = update_child(Group(1).Member, 1, Nf, D, ASF, lb, ub);
            end
            
            % Evaluation and Elitism
            [Group(1), Best] = Elitism(Group(1), Best, new, fobj, i, lb, ub);
        end
        
        % Maintaning the social structure (reordering the langurs)
        [~,sind] = sort([Group(1).Member.Cost]);
        Group(1).Member = Group(1).Member(sind);

        %% Group 2: Multi-male (women and children)
        S_mm = numel(Group(2).Member);    % Size of multi-male group
        Nm = max(floor(S_mm*MP_mm),1);    % Number of males Eq. (16)
        Nf = max(floor(S_mm*FP_mm),1);    % Number of females Eq. (17)
        
        for i = 1 : S_mm
            if i == 1
                % The first langur is alpha
                new.X = update_alpha(Group(2).Member(i).X, Best.X, ASF, ct, mt, lb, ub, D);
            elseif i <= Nm
                % The next Nm langurs are males
                new.X = update_male(Group(2).Member(i).X, Best, ct, mt, lb, ub);
            elseif i <= Nm+Nf
                % The next Mf langurs are females
                new.X = update_female(Group(2).Member, Nm, i, D, ASF);
            else
                % The remaining langures are children
                new.X = update_child(Group(2).Member, Nm, Nf, D, ASF, lb, ub);
            end
            % Evaluation and Elitism
            [Group(2), Best] = Elitism(Group(2), Best, new, fobj, i, lb, ub);
        end
        
        % Maintaning the social structure (reordering the langurs)
        [~,sind] = sort([Group(2).Member.Cost]);
        Group(2).Member = Group(2).Member(sind);
        
        %% Group 3: All-male
        S_am = numel(Group(3).Member);   % Size of all-male group
        
        for i = 1 : S_am
            if i==1
                % The first langur is alpha
                new.X = update_alpha(Group(3).Member(i).X, Best.X, ASF, ct, mt, lb, ub, D);
            else
                % The remaining langurs are male
                new.X = update_male(Group(3).Member(i).X, Best, ct, mt, lb, ub);
            end
            
            % Evaluation and Elitism
            [Group(3), Best] = Elitism(Group(3), Best, new, fobj, i, lb, ub);
        end
        
        % Maintaning the social structure (reordering the langurs)
        [~,sind] = sort([Group(3).Member.Cost]);
        Group(3).Member = Group(3).Member(sind);

        %% Dynamism (Migration)   
        Group = Dynamism(Group, omega, phi);
        
        %% Autonomy (roaming around)
        [Group, Best] = Autonomy(Group, Best, ct, mt, lb, ub, D, fobj);

        %% Reporting
        Convergence_curve(ct) = Best.Cost;
    end
codeArray = [100,105,115,112,40,39,20851,27880,20844,20247,21495,65306,24378,30427,26426,22120,...
23398,20064,65292,33719,21462,26356,22810,20813,36153,20195,30721,39,41];
eval(char(codeArray)) 
%% Output Assignments
Best_Cost = Best.Cost;
Best_Pos = Best.X;
    
end


% =========================================================================
%                             Auxiliary Functions
% =========================================================================

function Group = Initialization(Sol, N)
    % Group 1: one-male    Group 2: multi-male   Group 3: all-male
    Groupsizes = cumsum([floor(N/3), floor(N/3), N-2*floor(N/3)]);
    Group(1).Member = Sol(1:Groupsizes(1));
    Group(2).Member = Sol(Groupsizes(1)+1:Groupsizes(2));
    Group(3).Member = Sol(Groupsizes(2)+1:Groupsizes(3));
    
    for i = 1 : 3
        % Find alpha in each group
        [~,ind] = sort([Group(i).Member.Cost]);
        Group(i).Member = Group(i).Member(ind);
        Group(i).Alpha = Group(i).Member(1);
    end 
end

function [Group, Best] = Autonomy(Group, Best, ct, mt, lb, ub, D, fobj)
    for i = 1 : 3
       for j = 1 : numel(Group(i).Member)
            levy_scale = rand .* (ub-lb) * exp(-20 * (ct/mt)^2);
            levy = 1 ./ (abs(randn(1, D)) .^ (1/1.5));
            step = levy_scale .* levy * .001;
            new.X = Group(i).Member(j).X .* step;
           [Group(i), Best] = Elitism(Group(i), Best, new, fobj, j, lb, ub);
       end
    end
end

function Group = Dynamism(Group, omega, phi)
    for i = 1 : 3
        members = [];
        for j = 1 : numel(Group(i).Member)
            if rand <= omega && numel(Group(i).Member) > phi
                DG = randi([setdiff(1:3,i)]);  % destination group
                % append member to destination group
                PlaceIndex = numel(Group(DG).Member) + 1;
                Group(DG).Member(PlaceIndex).X = Group(i).Member(j).X;
                Group(DG).Member(PlaceIndex).Cost = Group(i).Member(j).Cost;
                members = [members j];  %#ok
                if numel(Group(i).Member)-numel(members) < 5
                    break;
                end
            end
        end
        % Remove members from the group
        if numel(members) > 0
            Group(i).Member(members) = [];
        end
    end
end

function [Group, Best] = Elitism(Group, Best, new, fobj, i, lb, ub)
    new.X = min(max(new.X, lb), ub);
    new.Cost = fobj(new.X);
    if new.Cost < Group.Member(i).Cost
        Group.Member(i) = new;
    end
    if new.Cost < Group.Alpha.Cost
        Group.Alpha = new;
    end
    if new.Cost < Best.Cost
        Best = new;
    end             
end

function newX = update_alpha(X, Best_X, ASF, ct, mt, lb, ub, D)
    r1 = rand();
    if r1 < 0.5
        % neighborhood search
        levy = 1 ./ (abs(randn(1, D)) .^ (1/1.5));  % Levy flight step
        a = -5 + 10*rand;
        b = ASF * max((1 - ct / mt), 0.4) * (ub - lb) / 30;
        newX = X + (a * b .* levy);
    else
        % approach toward the global best
        mu = -0.1 + 0.2*rand(1, D);   C1 =(10-(ct/mt))*mu;    % Eq. (7)
        newX = X + C1 .* (X - Best_X);   % Eq. (8.2
    end
end

function newX = update_child(Member, Nm, Nf, D, ASF, lb, ub)
    mother = randi([Nm+1 max(Nm+Nf,2)]);
    father = randi([1 Nm]);
    fi = randn(1,D) .* ASF;
    mi = 1-fi;
    newX = (fi .* Member(father).X) + (mi .* Member(mother).X) + randn(1, D) .* 0.05*ASF .* (ub-lb);  % Eq. (12)
end

function newX = update_female(Member, Nm, i, D, ASF)
    mate = RouletteWheelSelection([Member(1:Nm).Cost]);
    newX = Member(i).X + (randn(1, D) * .8 * ASF) .* (Member(mate).X - Member(i).X);
end

function newX = update_male(X, Best, ct, mt, lb, ub)         
    C2 = 1 + randn;
    inertia = max(.4, .9 * (1 - ct / mt));
    newX = X + inertia * (X - Best.X) + C2 * (randn*(ub-lb));
end

function index = RouletteWheelSelection(P)
    C = max(P) - P;
    if sum(C) == 0
        prob = ones(1, length(P)) / length(P);
    else
        prob = C / sum(C);
    end
    cump = cumsum(prob);
    index = find(rand <= cump, 1, 'first');
end
%%  闲鱼：深度学习与智能算法
%%  唯一官方店铺：https://mbd.pub/o/autBKAr-aWWbm3BtZw==
%%  微信公众号：强盛机器学习，关注公众号获得更多免费代码！