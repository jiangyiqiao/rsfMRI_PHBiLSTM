clear;
clc;
load('../data/fmri_dti_73.mat')
fmri_73 = fmri_dti_73(:,2);
for k=1:73
    fmri_73{k}=fmri_73{k}(:,1:90);
end
roi_datas=zeros(73,180,90);
for i=1:73
     roi_datas(i,:,:)=fmri_73{i};
end

NUM = 73;
FRAME = 180;
win_size = 68;
win_stride = 2;
win_num = (FRAME-win_size+win_stride)/win_stride-1;
fmri=zeros(FRAME,90,NUM);
low_net=zeros(90,90,NUM,win_num);
fmri_sliding=zeros(win_size+win_stride,90,NUM,win_num);
win = win_stride+win_size;
for i=1:NUM
    fmri(:,:,i)=reshape(roi_datas(i,:,:),FRAME,90,1);
end

for i=1:win_num
    fmri_sliding(:,:,:,i)=fmri(win_stride*i-1:win_stride*i+win_size,:,:);    %滑动窗口
end


for i=1:win_num
    for j=1:NUM
        low_net(:,:,j,i)=corr(fmri_sliding(:,:,j,i));      %构建低阶网络
    end
    disp(['建低阶网络已完成',num2str(i)]);
end

% 去除脑区相关
lamda = 0.01;
Folder_Original_Data = strcat('../data/');
Lassofile = strcat('nets_group_lasso_',num2str(lamda),'.mat');
load (fullfile(Folder_Original_Data,'/groupLasso',Lassofile));  % 加载GroupLasso文件

for i=1:NUM
    for p=1:90
        for q=1:90
            if abs(nets(p,q,i))< 0.01
                low_net(p,q,:,:)=0;
            else
                continue
            end
        end
    end
end

temp_net = reshape(low_net,8100,NUM*win_num);
temp_net(all(temp_net==0,2),:)=[];

feature_count = size(temp_net,1);
sliwinData = cell(NUM,4);
sliwin = permute(reshape(temp_net,feature_count,NUM,win_num),[2,3,1]);


for i =1:NUM
    temp = squeeze(sliwin(i,:,:));
    temp(:,all(temp==0,1))=[];
    sliwinData{i,1} = temp;
    sliwinData{i,2} = fmri_dti_73{i,1};
    sliwinData{i,3} = fmri_dti_73{i,2};
    sliwinData{i,4} = fmri_dti_73{i,3};
end

save(strcat('../data/features/sliwinCorr_',num2str(win_num),'.mat'),'sliwinData');