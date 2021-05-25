function  [nets] = net_built_group_lasso(fmri,lamda)
t=numel(fmri);
n=size(fmri{1},2);
nets=zeros(n,n,t);
opts = [];
opts.q = 2;
opts.init = 2;
opts.tFlag = 5;
opts.maxIter = 1000;
opts.nFlag = 0;
opts.rFlag = 1;
opts.ind = [];
for i = 1:n
    Y=[];
    X=[];
    for j= 1 : t
        temp0 = fmri{j};
        temp = temp0;
        Y = [Y ; temp(:,i)];
        temp(:,i)=0*temp(:,i);
        X = [X ;temp];
    end
    [MM,~]=size(Y);
    opts.ind =0:MM/t:MM;
    [x1, ~, ~] = mtLeastR(X ,Y,lamda,opts);
    nets(:,i,:)=x1;
    disp('***************************************************************');
    disp(['       ',num2str(100*i/n),'% of net construction is finished']);%, lamda_best is: ',num2str(lamda_best)]);
    disp('***************************************************************');
end
end

