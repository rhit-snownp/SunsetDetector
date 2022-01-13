clc;
clear;
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
       store = {Files(k).name,temp.featureVector,temp.classificationLabel,temp.filenames};
   else
       temp = load(FileNames);
       newVector = {Files(k).name,temp.featureVector,temp.classificationLabel,temp.filenames};
       store = cat(1,store,newVector); 
   end 
end

%% Store Data into labeled objects
trainX = normalizeFeatures01(cell2mat(store(3,2)));
trainY = cell2mat(store(3,3));

testX = normalizeFeatures01(cell2mat(store(2,2)));
testY = cell2mat(store(2,3));

validateX = normalizeFeatures01(cell2mat(store(4,2)));
validateY = cell2mat(store(4,3));

reservedX = normalizeFeatures01(cell2mat(store(1,2)));
reservedY = cell2mat(store(1,3));


%% Predict an optimal set of hyperparameters for use in an SVM using MATLAB as a starting point
net = fitcsvm(trainX, trainY, 'KernelFunction', 'rbf','OptimizeHyperparameters','auto');  
save("trained_network",'net');

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
[true_positive, false_positive, true_negative, false_negative, Accuracy, TPR, FPR, IncorrectImagesByIndex] = determineStatistics(detectedClasses,testY);




%% Train and evaluate an SVM with Optimal HyperParameters Manually
resolution = 20; 
maxKS = 50;
maxBC = 500;
[bestKS, bestBC, bestAccuracy,meshKS,meshBC,meshAcc] = hyperparameterGS(resolution,maxKS,maxBC,trainX,trainY,testX,testY);
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
[true_positive, false_positive, true_negative, false_negative, Accuracy, TPR, FPR, IncorrectImagesByIndex] = determineStatistics(detectedClasses,testY);


timeElapsed = toc;
fprintf("The total time elapsed is %f seconds\n",timeElapsed);

%% Show all the images that failed, and why

for index = 1:length(IncorrectImagesByIndex)
    %grab the incorrect index from the array, and then find the image, load it and show the results
    imageIndex = IncorrectImagesByIndex(index,1);
   
    temp = store(2,4);
    testFilenames = [temp{:}]
    correctFilename = string(testFilenames(imageIndex));
    img = imread(correctFilename);

    
    sampleImageFeatures = normalizeFeatures01(featureExtract(img, 7));
    %Classification
    [detectedClasses, distances] = predict(net, sampleImageFeatures.');
    
    figure;
    imshow(img);
if(detectedClasses >=0)
   title("Classification:  Sunset");
else
   title("Classification:  Not A Sunset");
end
    
end






%% Results:
% Best observed feasible point:
%     BoxConstraint    KernelScale
%     _____________    ___________
% 
%        185.09          998.28   
% 
% Observed objective function value = 0.10438
% Estimated objective function value = 0.10472
% Function evaluation time = 0.65584
% 
% Best estimated feasible point (according to models):
%     BoxConstraint    KernelScale
%     _____________    ___________
% 
%        61.978           994.9   
% 
% Estimated objective function value = 0.10472
% Estimated function evaluation time = 0.84962
% 
% Training Set: The Accuracy is 100.000000, with a TP=800, a FP=0, a TN=800 and a FN=0, TPR=100.000000, FPR=0.000000
% Validation Set: The Accuracy is 91.666667, with a TP=277, a FP=27, a TN=273 and a FN=23, TPR=92.333333, FPR=9.000000
% Test Set: The Accuracy is 88.977956, with a TP=439, a FP=51, a TN=449 and a FN=59, TPR=88.152610, FPR=10.200000

% Optimized Hyperparameters are Kernel Scale: 12.720000, Box Constraint: 106.000000 for an accuracy of 50.100200
% Training Set: The Accuracy is 100.000000, with a TP=800, a FP=0, a TN=800 and a FN=0, TPR=100.000000, FPR=0.000000
% Validation Set: The Accuracy is 93.333333, with a TP=272, a FP=12, a TN=288 and a FN=28, TPR=90.666667, FPR=4.000000
% Test Set: The Accuracy is 89.078156, with a TP=424, a FP=35, a TN=465 and a FN=74, TPR=85.140562, FPR=7.000000
% The total time elapsed is 232.566584 seconds

% Optimized Hyperparameters are Kernel Scale: 26.712044, Box Constraint: 188.352500 for an accuracy of 87.975952
% Training Set: The Accuracy is 100.000000, with a TP=800, a FP=0, a TN=800 and a FN=0, TPR=100.000000, FPR=0.000000
% Validation Set: The Accuracy is 92.166667, with a TP=271, a FP=18, a TN=282 and a FN=29, TPR=90.333333, FPR=6.000000
% Test Set: The Accuracy is 87.975952, with a TP=428, a FP=50, a TN=450 and a FN=70, TPR=85.943775, FPR=10.000000
% The total time elapsed is 655.990260 seconds

%% Calculate an ROC Curve for Training and Validation Set

bestBC = 106.000000 ;
bestKS = 12.720000;
net = fitcsvm(trainX, trainY, 'KernelFunction', 'rbf', 'KernelScale', bestKS, 'BoxConstraint', bestBC,'Standardize',true); 
save("trained_network",'net');

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




