clear all;
clc;   

trainPath='D:\Biomassdata\train_xy_113.xlsx';
testPath='D:\Biomassdata\test_xy_113.xlsx';

TrainData = xlsread(trainPath);
TestData = xlsread(testPath);

Train_c=TrainData(:,4:end); 
Test_c=TestData(:,4:end);

Train_num=TrainData(:,1:4);
Test_num=TestData(:,1:4);
numtree=500;

RF_mtry(Train_c,Test_c,'D:\Biomassdata\results\',Train_num,Test_num,numtree)
