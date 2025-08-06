% 清空环境
clear; clc;

% ====== 参数设置 ======
savePath = 'D:\Biomassdata\results\';  % 保存路径，可根据需要更改RF_transportability
mtry = 10;                         % 使用的特征数量

% ====== 加载模型和特征排序 ======
load([savePath, 'RFmodel_113.mat']);             % 加载模型
load([savePath, 'importance_index_113.mat']);    % 加载特征排序索引,这里的 index 是从 有叶模型中提取出来的特征重要性排名，保持一致即可

% ====== 加载测试数据 ======
testData = xlsread('D:\Biomassdata\test_xy_59.xlsx');

% 特征 + 标签区域（从第4列开始）
test_c = testData(:, 4:end);

% 使用模型预测:用已经训练好的“有叶模型”中选出的前10个重要特征的“列索引”来从“无叶数据”中提取相同的特征列，然后送进模型做预测
Classtest = regRF_predict(test_c(:, index(1:mtry,1)), RFmodel);

% ====== 精度评估 ======
R2_test = calR2(Classtest, test_c(:, end));
RMSE_test = calRMSE(Classtest, test_c(:, end));
rRMSE_test = RMSE_test / mean(test_c(:, end));

% 显示结果
fprintf('==============================\n');
fprintf('测试集精度评估结果：\n');
fprintf('R?       = %.4f\n', R2_test);
fprintf('RMSE     = %.4f\n', RMSE_test);
fprintf('rRMSE    = %.4f\n', rRMSE_test);
fprintf('==============================\n');

% ====== 保存预测结果 ======
xlswrite([savePath, '113model_transportability.xlsx'], [testData(:,1:3), test_c(:, end), Classtest]);

% 保存评估指标
R2_RMSE = [mtry, R2_test, RMSE_test, rRMSE_test]; % 可添加更多字段
xlswrite([savePath, 'R2_RMSE_113model_transportability.xlsx'], R2_RMSE);
