function RF_mtry(train,test,savePath,Train_num,Test_num,numtree)
%% 参数说明
% train:输入的训练数据，包含标签（标签位于最后一列）
% test:输入的测试数据，包含标签（标签位于最后一列）
% savePath：保存数据的路径，类似：'C:\'
%%
addpath(genpath('./Toolboxs/'));  %加入分类调用的相关函数路径
data=[train;test];

nTree=100;
nLeaf=20;
RFModel=TreeBagger(nTree,train(:,1:end-1),train(:,end), 'Method','regression','OOBPredictorImportance','on','MinLeafSize',nLeaf);%?????????

RFModel.OOBPermutedPredictorDeltaError

temp=[1:length(RFModel.OOBPermutedPredictorDeltaError);RFModel.OOBPermutedPredictorDeltaError']';
index=sortrows(temp,-2);
%%
R2_RMSE=[];

mtry=5 %直接取index 的前5个的变量号索引

RFModel=TreeBagger(nTree,train(:,index(1:mtry,1),train(:,end),'Method','regression','OOBPredictorImportance','on','MinLeafSize',nLeaf);%?????????

[RFPredictYield,RFPredictConfidenceInterval]=predict(RFModel,train(:,index(1:mtry,1)));

    % train
Classtrain = RFPredictYield
R2_train=calR2(Classtrain,train(:,end));%R2
RMSE_train=calRMSE(Classtrain,train(:,end));%RMSE
rRMSE_train=RMSE_train/mean(train(:,end));%RMSE
    
        
    % test
Classtest = predict(RFmodel,test(:,index(1:mtry,1)));%用得到的模型去预测测试数据(即整个数据)
R2_test=calR2(Classtest,test(:,end));%R2
RMSE_test=calRMSE(Classtest,test(:,end));%RMSE
rRMSE_test=RMSE_test/mean(test(:,end));%RMSE
    
    % data
Classdata = predict(RFmodel,data(:,index(1:mtry,1)));%用得到的模型去预测全部的数据(即整个数据)
R2_data=calR2(Classdata,data(:,end));%R2
RMSE_data=calRMSE(Classdata,data(:,end));%RMSE
rRMSE_data=RMSE_data/mean(data(:,end));%RMSE 
        
R2_RMSE=[R2_RMSE;[mtry,R2_train,RMSE_train,rRMSE_train,R2_test,RMSE_test,rRMSE_test,R2_data,RMSE_data,rRMSE_data]];
xlswrite([savePath,'train_predict_',num2str(mtry),'.xlsx'],[train(:,end),Classtrain])
    %xlswrite([savePath,'train_predict_xy_',num2str(mtry),'.xlsx'],[Train_num(:,1:3),train(:,end),Classtrain])%在训练数据上添加经纬度
xlswrite([savePath,'test_predict_',num2str(mtry),'.xlsx'],[test(:,end),Classtest])
    %xlswrite([savePath,'test_predict_xy_',num2str(mtry),'.xlsx'],[Test_num(:,1:3),test(:,end),Classtest])%在测试数据上添加经纬度
xlswrite([savePath,'data_predict_',num2str(mtry),'.xlsx'],[data(:,end),Classdata])

xlswrite([savePath,'numtree.xlsx'],numtree)
xlswrite([savePath,'train.xlsx'],train)
xlswrite([savePath,'train_xy.xlsx'],[Train_num(:,1:3),train])
xlswrite([savePath,'test.xlsx'],test)
xlswrite([savePath,'test_xy.xlsx'],[Test_num(:,1:3),test])
xlswrite([savePath,'R2_RMSE.xlsx'],R2_RMSE)
xlswrite([savePath,'impotance.xlsx'],index)
end