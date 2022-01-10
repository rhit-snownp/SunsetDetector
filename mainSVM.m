clc;
clear all;
close all;

tic;

%% Load Data into the Script Memory
currentDirectory = pwd;
cd ('featureVectors');
Files = dir('*.mat');
cd(currentDirectory); 


%Files=dir('FeatureVectors');
store = []; 
for k=1:length(Files)
   FileNames=Files(k).name;
   %newStr = extractAfter(str,12)
   
   if(isempty(store))
       temp = load(FileNames);
       store = {Files(k).name,temp.featureVector,temp.classificationLabel};
   else
       temp = load(FileNames);
       newVector = {Files(k).name,temp.featureVector,temp.classificationLabel};
       store = cat(1,store,newVector); 
   end 
end

%% Store Data into labeled objects
trainX = cell2mat(store(3,2));
trainY = cell2mat(store(3,3));

testX = cell2mat(store(2,2));
testY = cell2mat(store(2,3));

validateX = cell2mat(store(4,2));
validateY = cell2mat(store(4,3));

reservedX = cell2mat(store(1,2));
reservedY = cell2mat(store(1,3));


%% Predict an optimal set of hyperparameters for use in an SVM using MATLAB as a starting point


net = fitcsvm(trainX, trainY, 'KernelFunction', 'rbf','OptimizeHyperparameters','auto'); 



%Classifies Training Set
[detectedClasses, distances] = predict(net, trainX);

%Calculate Statistics
fprintf("Training Set: ");
[true_positive, false_positive, true_negative, false_negative, Accuracy, TPR, FPR] = determineStatistics(detectedClasses,trainY);

%Classifies Validation Set
[detectedClasses, distances] = predict(net, validateX);

%Calculate Statistics
fprintf("Validation Set: ");
[true_positive, false_positive, true_negative, false_negative, Accuracy, TPR, FPR] = determineStatistics(detectedClasses,validateY);

%Classifies Testing Set
[detectedClasses, distances] = predict(net, testX);

%Calculate Statistics
fprintf("Test Set: ");
[true_positive, false_positive, true_negative, false_negative, Accuracy, TPR, FPR] = determineStatistics(detectedClasses,testY);






%% Train and evaluate an SVM with Optimal HyperParameters Manually



resolution = 20; 
maxKS = 30;
maxBC = 250;
[bestKS, bestBC, bestAccuracy] = hyperparameterGS(resolution,maxKS,maxBC,trainX,trainY,testX,testY);
fprintf("Optimized Hyperparameters are Kernel Scale: %f, Box Constraint: %f for an accuracy of %f\n", bestKS, bestBC, bestAccuracy * 100);
net = fitcsvm(trainX, trainY, 'KernelFunction', 'rbf', 'KernelScale', bestKS, 'BoxConstraint', bestBC,'Standardize',true); 

%Classifies Training Set
[detectedClasses, distances] = predict(net, trainX);

%Calculate Statistics
fprintf("Training Set: ");
[true_positive, false_positive, true_negative, false_negative, Accuracy, TPR, FPR] = determineStatistics(detectedClasses,trainY);

%Classifies Validation Set
[detectedClasses, distances] = predict(net, validateX);

%Calculate Statistics
fprintf("Validation Set: ");
[true_positive, false_positive, true_negative, false_negative, Accuracy, TPR, FPR] = determineStatistics(detectedClasses,validateY);

%Classifies Testing Set
[detectedClasses, distances] = predict(net, testX);

%Calculate Statistics
fprintf("Test Set: ");
[true_positive, false_positive, true_negative, false_negative, Accuracy, TPR, FPR] = determineStatistics(detectedClasses,testY);



timeElapsed = toc;
fprintf("The total time elapsed is %f seconds\n",timeElapsed);


%% Results:

% Best estimated feasible point (according to models):
%     BoxConstraint    KernelScale
%     _____________    ___________
% 
%        277.89            886    
% 
% Estimated objective function value = 0.11167
% Estimated function evaluation time = 0.74368
% 
% Training Set: The Accuracy is 100.000000, with a TP=800, a FP=0, a TN=800 and a FN=0, TPR=100.000000, FPR=0.000000
% Validation Set: The Accuracy is 91.500000, with a TP=274, a FP=25, a TN=275 and a FN=26, TPR=91.333333, FPR=8.333333
% Test Set: The Accuracy is 88.777555, with a TP=435, a FP=49, a TN=451 and a FN=63, TPR=87.349398, FPR=9.800000
% 
% Optimized Hyperparameters are Kernel Scale: 12.720000, Box Constraint: 106.000000 for an accuracy of 50.100200
% Training Set: The Accuracy is 100.000000, with a TP=800, a FP=0, a TN=800 and a FN=0, TPR=100.000000, FPR=0.000000
% Validation Set: The Accuracy is 93.333333, with a TP=272, a FP=12, a TN=288 and a FN=28, TPR=90.666667, FPR=4.000000
% Test Set: The Accuracy is 89.078156, with a TP=424, a FP=35, a TN=465 and a FN=74, TPR=85.140562, FPR=7.000000



%% Calculate an ROC Curve for Training and Validation Set

bestBC = 106.000000 ;
bestKS = 12.720000;
net = fitcsvm(trainX, trainY, 'KernelFunction', 'rbf', 'KernelScale', bestKS, 'BoxConstraint', bestBC,'Standardize',true); 

% ROC Curve for training set
[~, distances] = predict(net, trainX);

numPoints = 10000;
thresholdValues = linspace(-1,1,numPoints);
TruePositiveRateArray = ones(numPoints,1);
FalsePositiveRateArray = ones(numPoints,1);

for currentPoint = 1:numPoints
    threshold = thresholdValues(1,currentPoint);
    detectedClasses =  double(distances(:,2) >= threshold);
    detectedClasses(detectedClasses == 0) = -1;
    [~, ~, ~, ~, ~, TruePositiveRateArray(currentPoint,1), FalsePositiveRateArray(currentPoint,1)] = determineStatistics(detectedClasses,testY);
end


[figureHandle] = generateROC(TruePositiveRateArray,FalsePositiveRateArray,"ROC Curve for Training Set");


% ROC Curve for validation set
[~, distances] = predict(net, validateX);

numPoints = 10000;
thresholdValues = linspace(-1,1,numPoints);
TruePositiveRateArray = ones(numPoints,1);
FalsePositiveRateArray = ones(numPoints,1);

for currentPoint = 1:numPoints
    threshold = thresholdValues(1,currentPoint);
    detectedClasses =  double(distances(:,2) >= threshold);
    detectedClasses(detectedClasses == 0) = -1;
    [~, ~, ~, ~, ~, TruePositiveRateArray(currentPoint,1), FalsePositiveRateArray(currentPoint,1)] = determineStatistics(detectedClasses,validateY);
end


[figureHandle] = generateROC(TruePositiveRateArray,FalsePositiveRateArray,"ROC Curve for Validation Set");


% ROC Curve for Test set
[~, distances] = predict(net, testX);

numPoints = 10000;
thresholdValues = linspace(-1,1,numPoints);
TruePositiveRateArray = ones(numPoints,1);
FalsePositiveRateArray = ones(numPoints,1);

for currentPoint = 1:numPoints
    threshold = thresholdValues(1,currentPoint);
    detectedClasses =  double(distances(:,2) >= threshold);
    detectedClasses(detectedClasses == 0) = -1;
    [~, ~, ~, ~, ~, TruePositiveRateArray(currentPoint,1), FalsePositiveRateArray(currentPoint,1)] = determineStatistics(detectedClasses,testY);
end


[figureHandle] = generateROC(TruePositiveRateArray,FalsePositiveRateArray,"ROC Curve for Test Set");




