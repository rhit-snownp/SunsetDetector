%  CNN for feature extraction. 
%  The next option is to use the feature extraction part of one of the pre-trained CNNs 
%  and then use the features extracted for each image as inputs for an SVM (one different than the one from Part 1, 
%  so must be trained and tuned in the same fashion as that one), and a new ROC curve generated.


clc;
clear;
close all;
tic;

%% Create Datastores for Each Dataset
filepath = "..\\images\train";
trainingDataStore = imageDatastore(filepath,'IncludeSubfolders',true,'LabelSource','foldernames');


filepath = "..\\images\test";
testingDataStore = imageDatastore(filepath,'IncludeSubfolders',true,'LabelSource','foldernames');


filepath = "..\\images\validate";
validationDataStore = imageDatastore(filepath,'IncludeSubfolders',true,'LabelSource','foldernames');


%% Load in Pre-Trained CNN
net = alexnet;
net.Layers;
inputSize = net.Layers(1).InputSize;

%% Create the augmented datastores with the correct input size
augmentedTrain = augmentedImageDatastore(inputSize(1:2),trainingDataStore);
augmentedTest = augmentedImageDatastore(inputSize(1:2),testingDataStore);
augmentedValidation = augmentedImageDatastore(inputSize(1:2),validationDataStore);

%% Generate Features from the activation layer for each dataset
layer = 'fc7';
featuresTrain = activations(net,augmentedTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augmentedTest,layer,'OutputAs','rows');
featuresValidation = activations(net,augmentedValidation,layer,'OutputAs','rows');

%% Extract the class labels from the training and test data.
tempLableArray = zeros(length(trainingDataStore.Labels),1);
tempLableArray(trainingDataStore.Labels == 'sunset') = 1;
tempLableArray(trainingDataStore.Labels == 'nonsunset') = -1;
trainingDataStore.Labels = tempLableArray;

tempLableArray = zeros(length(testingDataStore.Labels),1);
tempLableArray(testingDataStore.Labels == 'sunset') = 1;
tempLableArray(testingDataStore.Labels == 'nonsunset') = -1;
testingDataStore.Labels = tempLableArray;

tempLableArray = zeros(length(validationDataStore.Labels),1);
tempLableArray(validationDataStore.Labels == 'sunset') = 1;
tempLableArray(validationDataStore.Labels == 'nonsunset') = -1;
validationDataStore.Labels = tempLableArray;

YTrain = trainingDataStore.Labels;
YTest = testingDataStore.Labels;
YValidate = validationDataStore.Labels;

%% Fit an SVM to the Training Data
[SVMmodel,HyperparameterOptimizationResults] = fitcecoc(featuresTrain,YTrain);

%Predict the Results
[YPredTest, distancesTest] = predict(SVMmodel,featuresTest);
accuracyTest = mean(YPredTest == YTest);

[YPredTrain, distancesTrain] = predict(SVMmodel,featuresTrain);
accuracyTrain = mean(YPredTrain == YTrain);

[YPredValidate, distancesValidate] = predict(SVMmodel,featuresValidation);
accuracyValidate = mean(YPredValidate == YValidate);

fprintf("Testing Accuracy: %f, Training Accuracy %f, Validation Accuracy %f\n",accuracyTest*100,accuracyTrain*100,accuracyValidate*100);

elapsedTime = toc;
fprintf("The total time elapsed is %f seconds\n",elapsedTime);



%% Generate all of the ROC Curves
% ROC Curve for Test set
numPoints = 100;
thresholdValues = linspace(-1,1,numPoints);
TruePositiveRateArray = ones(numPoints,1);
FalsePositiveRateArray = ones(numPoints,1);


for currentPoint = 1:numPoints
    threshold = thresholdValues(1,currentPoint);
    detectedClasses =  double(distancesTest(:,2) >= threshold);
    detectedClasses(detectedClasses == 0) = -1;
    [~, ~, ~, ~, ~, TruePositiveRateArray(currentPoint,1), FalsePositiveRateArray(currentPoint,1)] = determineStatistics(detectedClasses, distancesTest, YTest);
end


[figureHandle] = generateROC(TruePositiveRateArray,FalsePositiveRateArray,"ROC Curve for Test Set");


%%
% ROC Curve for Train set
numPoints = 100;
thresholdValues = linspace(-1,1,numPoints);
TruePositiveRateArray = ones(numPoints,1);
FalsePositiveRateArray = ones(numPoints,1);

for currentPoint = 1:numPoints
    threshold = thresholdValues(1,currentPoint);
    detectedClasses =  double(distancesTrain(:,2) >= threshold);
    detectedClasses(detectedClasses == 0) = -1;
    [~, ~, ~, ~, ~, TruePositiveRateArray(currentPoint,1), FalsePositiveRateArray(currentPoint,1)] = determineStatistics(detectedClasses, distancesTrain, YTrain);
end


[figureHandle] = generateROC(TruePositiveRateArray,FalsePositiveRateArray,"ROC Curve for Train Set");

%%
% ROC Curve for Validation set
numPoints = 100;
thresholdValues = linspace(-1,1,numPoints);
TruePositiveRateArray = ones(numPoints,1);
FalsePositiveRateArray = ones(numPoints,1);

for currentPoint = 1:numPoints
    threshold = thresholdValues(1,currentPoint);
    detectedClasses =  double(distancesValidate(:,2) >= threshold);
    detectedClasses(detectedClasses == 0) = -1;
    [~, ~, ~, ~, ~, TruePositiveRateArray(currentPoint,1), FalsePositiveRateArray(currentPoint,1)] = determineStatistics(detectedClasses, distancesValidate, YValidate);
end


[figureHandle] = generateROC(TruePositiveRateArray,FalsePositiveRateArray,"ROC Curve for Validation Set");