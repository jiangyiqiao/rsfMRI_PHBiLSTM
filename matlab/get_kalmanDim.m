clear;
clc;
savefilepath = '../data/features/';
for adg = [0.9]
% for adg=0.1:0.1:0.9
    load(strcat('../data/kalman/kalmancorr_0.01_',num2str(adg),'.mat'));
    corr1=zeros(13140,8100);
    for i=1:73
        for j=1:90
            for k=1:90
                corr1(180*i-179:180*i,j*90+k-90)=squeeze(corr{i,1}(j,k,:));
            end
        end
    end

    % ÌØÕ÷Êý
    num = sum(corr1(4,:)~=0);
    [m,n] = size(corr1);
    corr2=zeros(13140,num);
    j = 1;
    for i=1:8100
        if length(find(corr1(:,i))==0)<13140
            continue
        else
            corr2(:,j)=corr1(:,i);
            j = j + 1;
        end
    end

    datas = cell(73,4);
    for i=1:73
        datas{i,1}=corr2(180*i-179:180*i,:);
        datas{i,2}=corr{i,2};
        datas{i,3}=corr{i,3};
        datas{i,4}=corr{i,4};
    end
    save([savefilepath,strcat('kalmancorr_0.01_',num2str(adg),'_',num2str(num),'.mat')],'datas');

end
