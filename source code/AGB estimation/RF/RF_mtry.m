
function RF_mtry(train,test,savePath,Train_num,Test_num,numtree)
addpath(genpath('D:\Biomassdata\随机森林回归代码\Toolboxs'));  

RFmodel = regRF_train(train(:,1:end-1),train(:,end),numtree);
temp=[1:length(RFmodel.importance);RFmodel.importance']';
index=sortrows(temp,-2);

R2_RMSE=[];

mtry = 10;
RFmodel = regRF_train(train(:,index(1:mtry,1)),train(:,end),numtree);%mtry=4

    % train
Classtrain = regRF_predict(train(:,index(1:mtry,1)),RFmodel);
R2_train=calR2(Classtrain,train(:,end));%R2
RMSE_train=calRMSE(Classtrain,train(:,end));%RMSE
rRMSE_train=RMSE_train/mean(train(:,end));%RMSE
            
    % test
Classtest = regRF_predict(test(:,index(1:mtry,1)),RFmodel);
R2_test=calR2(Classtest,test(:,end));%R2
RMSE_test=calRMSE(Classtest,test(:,end));%RMSE
rRMSE_test=RMSE_test/mean(test(:,end));%RMSE

R2_RMSE=[R2_RMSE;[mtry,R2_train,RMSE_train,rRMSE_train,R2_test,RMSE_test,rRMSE_test,]];

xlswrite([savePath,'train_predict_xy_113_',num2str(mtry),'.xlsx'],[Train_num(:,1:3),train(:,end),Classtrain])%在训练数据上添加经纬度
xlswrite([savePath,'test_predict_xy_113_',num2str(mtry),'.xlsx'],[Test_num(:,1:3),test(:,end),Classtest])%在测试数据上添加经纬度
xlswrite([savePath,'numtree_113.xlsx'],numtree)
xlswrite([savePath,'test_xy_113.xlsx'],[Test_num(:,1:3),test])
xlswrite([savePath,'R2_RMSE_113.xlsx'],R2_RMSE)
xlswrite([savePath,'impotance_113.xlsx'],index)

% 保存模型
save([savePath, 'RFmodel_113.mat'], 'RFmodel');

% 保存特征排序结果（index）
save([savePath, 'importance_index_113.mat'], 'index');

end