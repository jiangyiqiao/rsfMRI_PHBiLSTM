clc;clear;
%% 73 sample dataset
Folder_Original_Data = '../data/';
load (fullfile(Folder_Original_Data,'fmri_dti_73.mat'));
fmri_73 = fmri_dti_73(:,2);
for k=1:73
    fmri_73{k}=fmri_73{k}(:,1:90);
end
Labels = fmri_dti_73(:,3);
 
%%  group Lasso
for lamda = [0.01]
     Folder = '../data/groupLasso';  
     nets = net_built_group_lasso(fmri_73,lamda);
     save(fullfile(Folder,['nets_group_lasso_',num2str(lamda),'.mat']),'nets','Labels');
 end



