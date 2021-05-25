clc;
clear;


filename =  '../data/fmri_dti_73.mat';
savefilepath = '../data/kalman/';

load(filename);
load ('../data/groupLasso/nets_group_lasso_0.01.mat');

roi_data = fmri_dti_73(:,1:3);
[M,N] = size(roi_data);  % M=73,N=3

%% The kth row of thm contains the parameters associated with time k; 
% that is, the estimate parameters are based on the data in rows up to and including row k in z. 
% nn = [na nb nk] na and nb are the orders of the ARX model, and nk is the delay.nb and nk are specified as row vectors of length equal to number of inputs
% R1 as covariance matrix of the parameter changes per time step

for R1=[0.9]
    relation = zeros(90,90,180);
    corr = cell(M,N+1);
    for k=1:M
        for i=1:90
            for j=1:90
                if abs(nets(i,j,k))< 0.1
    %                 disp('为零脑区，脑区间无联系');
    %                 disp(nets(r,c,num))
    %                 disp([num2str(r),' ',num2str(c),' ',num2str(num)]);
    %                 count = count + 1; 
                    continue;
                else
                    temp =  roi_data{k,2};
                    % 输入 第一个脑区187个时间点
                    input = temp(:,i);
                    % 输出 另一个脑区187个时间点
                    output = temp(:,j);
                    z = [output input];
                    nn = [1 0 1];
                    
                    adg = R1;
                    [thm,yhat,P,phi] = rarx(z,nn,'kf',adg);
                    
                    relation(i,j,:)=thm;  
                end

            end
        end
        corr{k,1}=relation;
        corr{k,2}=roi_data{k,1};
        corr{k,3}=roi_data{k,2};
        corr{k,4}=roi_data{k,3};
    end
    savefilename = strcat('kalmancorr_0.01_',num2str(adg),'.mat');
    save([savefilepath,savefilename],'corr');
end




