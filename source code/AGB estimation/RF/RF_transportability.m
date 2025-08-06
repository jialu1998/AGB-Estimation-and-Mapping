% ��ջ���
clear; clc;

% ====== �������� ======
savePath = 'D:\Biomassdata\results\';  % ����·�����ɸ�����Ҫ����RF_transportability
mtry = 10;                         % ʹ�õ���������

% ====== ����ģ�ͺ��������� ======
load([savePath, 'RFmodel_113.mat']);             % ����ģ��
load([savePath, 'importance_index_113.mat']);    % ����������������,����� index �Ǵ� ��Ҷģ������ȡ������������Ҫ������������һ�¼���

% ====== ���ز������� ======
testData = xlsread('D:\Biomassdata\test_xy_59.xlsx');

% ���� + ��ǩ���򣨴ӵ�4�п�ʼ��
test_c = testData(:, 4:end);

% ʹ��ģ��Ԥ��:���Ѿ�ѵ���õġ���Ҷģ�͡���ѡ����ǰ10����Ҫ�����ġ������������ӡ���Ҷ���ݡ�����ȡ��ͬ�������У�Ȼ���ͽ�ģ����Ԥ��
Classtest = regRF_predict(test_c(:, index(1:mtry,1)), RFmodel);

% ====== �������� ======
R2_test = calR2(Classtest, test_c(:, end));
RMSE_test = calRMSE(Classtest, test_c(:, end));
rRMSE_test = RMSE_test / mean(test_c(:, end));

% ��ʾ���
fprintf('==============================\n');
fprintf('���Լ��������������\n');
fprintf('R?       = %.4f\n', R2_test);
fprintf('RMSE     = %.4f\n', RMSE_test);
fprintf('rRMSE    = %.4f\n', rRMSE_test);
fprintf('==============================\n');

% ====== ����Ԥ���� ======
xlswrite([savePath, '113model_transportability.xlsx'], [testData(:,1:3), test_c(:, end), Classtest]);

% ��������ָ��
R2_RMSE = [mtry, R2_test, RMSE_test, rRMSE_test]; % ����Ӹ����ֶ�
xlswrite([savePath, 'R2_RMSE_113model_transportability.xlsx'], R2_RMSE);
