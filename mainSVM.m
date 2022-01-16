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
net = fitcsvm(trainX, trainY, 'KernelFunction', 'rbf');%,'OptimizeHyperparameters','auto');  
%save("trained_network",'net');


%Classifies Training Set
[detectedClasses, distances] = predict(net, trainX);

%Calculate Statistics
fprintf("Training Set: ");
[true_positive, false_positive, true_negative, false_negative, Accuracy, TPR, FPR] = determineStatistics(detectedClasses, distances, trainY);

%Classifies Validation Set
[detectedClasses, distances] = predict(net, validateX);

%Calculate Statistics
fprintf("Validation Set: ");
[true_positive, false_positive, true_negative, false_negative, Accuracy, TPR, FPR] = determineStatistics(detectedClasses, distances, validateY);

%Classifies Testing Set
[detectedClasses, distances] = predict(net, testX);

%Calculate Statistics
fprintf("Test Set: ");
[true_positive, false_positive, true_negative, false_negative, Accuracy, TPR, FPR] = determineStatistics(detectedClasses, distances, testY);



% %% Determining Best and Worst Image in Each Category:
% 
% %True Positives:
% titleString = "True Positives:";
% %Far From Margin
% [~, maxImageIndex] = max(abs(true_positive(:,2)));
% maxImageGlobalIndex = true_positive(maxImageIndex,1);
% classification = detectedClasses(maxImageGlobalIndex);
% distance = distances(maxImageGlobalIndex);
% getImageFromIndex(store,net,maxImageGlobalIndex,titleString,classification,distance);
% 
% %Close To Margin
% [~, minImageIndex] = min(abs(true_positive(:,2)));
% minImageGlobalIndex =true_positive(minImageIndex,1);
% classification = detectedClasses(minImageGlobalIndex);
% distance = distances(minImageGlobalIndex);
% getImageFromIndex(store,net,minImageGlobalIndex,titleString,classification,distance);
% 
% %True Negatives:
% titleString = "True Negatives:";
% %Far From Margin
% [~, maxImageIndex] = max(abs(true_negative(:,2)));
% maxImageGlobalIndex =true_negative(maxImageIndex,1);
% classification = detectedClasses(maxImageGlobalIndex);
% distance = distances(maxImageGlobalIndex);
% getImageFromIndex(store,net,maxImageGlobalIndex,titleString,classification,distance);
% 
% %Close To Margin
% [~, minImageIndex] = min(abs(true_negative(:,2)));
% minImageGlobalIndex =true_negative(minImageIndex,1);
% classification = detectedClasses(maxImageGlobalIndex);
% distance = distances(minImageGlobalIndex);
% getImageFromIndex(store,net,minImageGlobalIndex,titleString,classification,distance);
% 
% %False Positives:
% titleString = "False Positives:";
% %Far From Margin
% [~, maxImageIndex] = max(abs(false_positive(:,2)));
% maxImageGlobalIndex =false_positive(maxImageIndex,1);
% classification = detectedClasses(maxImageGlobalIndex);
% distance = distances(maxImageGlobalIndex);
% getImageFromIndex(store,net,maxImageGlobalIndex,titleString,classification,distance);
% 
% %Close To Margin
% [~, minImageIndex] = min(abs(false_positive(:,2)));
% minImageGlobalIndex =false_positive(minImageIndex,1);
% classification = detectedClasses(minImageGlobalIndex);
% distance = distances(minImageGlobalIndex);
% getImageFromIndex(store,net,minImageGlobalIndex,titleString,classification,distance);
% 
% %False Negatives:
% titleString = "False Negatives:";
% %Far From Margin
% [~, maxImageIndex] = max(abs(false_negative(:,2)));
% maxImageGlobalIndex =false_negative(maxImageIndex,1);
% classification = detectedClasses(maxImageGlobalIndex);
% distance = distances(maxImageGlobalIndex);
% getImageFromIndex(store,net,maxImageGlobalIndex,titleString,classification,distance);
% 
% %Close To Margin
% [~, minImageIndex] = min(abs(false_negative(:,2)));
% minImageGlobalIndex =false_negative(minImageIndex,1);
% classification = detectedClasses(minImageGlobalIndex);
% distance = distances(minImageGlobalIndex);
% getImageFromIndex(store,net,minImageGlobalIndex,titleString,classification,distance);
%     
% 



%% Train and evaluate an SVM with Optimal HyperParameters Manually
resolution = 100; 
maxKS = 200;
maxBC = 200;
minKS = 0;
minBC = 0;

[bestKS, bestBC, bestAccuracy,meshKS,meshBC,meshAcc] = hyperparameterGS(resolution,maxKS,maxBC,trainX,trainY,testX,testY);

%[bestKS, bestBC, bestAccuracy,meshKS,meshBC,meshAcc,numSupportVectors] = optimalGridSearch(resolution,minKS,maxKS,minBC,maxBC,trainX,trainY,testX,testY);
net = fitcsvm(trainX, trainY, 'KernelFunction', 'rbf', 'KernelScale', bestKS, 'BoxConstraint', bestBC,'Standardize',true); 
fprintf("Optimized Hyperparameters are Kernel Scale: %f, Box Constraint: %f for an accuracy of %f with %d support vectors\n", bestKS, bestBC, bestAccuracy * 100, length(net.SupportVectorLabels));
%save("trained_network",'net');

%Classifies Training Set
[detectedClasses, distances] = predict(net, trainX);

%Calculate Statistics
fprintf("Training Set: ");
[true_positive, false_positive, true_negative, false_negative, Accuracy, TPR, FPR] = determineStatistics(detectedClasses, distances, trainY);

%Classifies Validation Set
[detectedClasses, distances] = predict(net, validateX);

%Calculate Statistics
fprintf("Validation Set: ");
[true_positive, false_positive, true_negative, false_negative, Accuracy, TPR, FPR] = determineStatistics(detectedClasses, distances, validateY);

%Classifies Testing Set
[detectedClasses, distances] = predict(net, testX);

%Calculate Statistics
fprintf("Test Set: ");
[true_positive, false_positive, true_negative, false_negative, Accuracy, TPR, FPR] = determineStatistics(detectedClasses, distances, testY);


timeElapsed = toc;
fprintf("The total time elapsed is %f seconds\n",timeElapsed);


%%
figure
scatter3(meshKS,meshBC,meshAcc)

[xq,yq] = meshgrid(0:0.1:maxKS, 0:0.1:maxBC);
vq = griddata(meshKS,meshBC,meshAcc,xq,yq);  %(x,y,v) being your original data for plotting points
figure
mesh(xq,yq,vq)
hold on
plot3(meshKS,meshBC,meshAcc,'.','Color','k')
hold off

title("Plot of Optimized Hyperparameters");
xlabel("Kernel Scale");
ylabel("Box Constraint");
zlabel("Accuracy");


%% Calculate an ROC Curve for Training and Validation Set

bestBC = 106.000000 ;
bestKS = 12.720000;
net = fitcsvm(trainX, trainY, 'KernelFunction', 'rbf', 'KernelScale', bestKS, 'BoxConstraint', bestBC,'Standardize',true); 
fprintf("Optimized Hyperparameters are Kernel Scale: %f, Box Constraint: %f with %d support vectors\n", bestKS, bestBC, length(net.SupportVectorLabels));

%save("trained_network",'net');


%Classifies Training Set
[detectedClasses, distances] = predict(net, trainX);

%Calculate Statistics
fprintf("Training Set: ");
[true_positive, false_positive, true_negative, false_negative, Accuracy, TPR, FPR] = determineStatistics(detectedClasses, distances, trainY);

%Classifies Validation Set
[detectedClasses, distances] = predict(net, validateX);

%Calculate Statistics
fprintf("Validation Set: ");
[true_positive, false_positive, true_negative, false_negative, Accuracy, TPR, FPR] = determineStatistics(detectedClasses, distances, validateY);

%Classifies Testing Set
[detectedClasses, distances] = predict(net, testX);

%Calculate Statistics
fprintf("Test Set: ");
[true_positive, false_positive, true_negative, false_negative, Accuracy, TPR, FPR] = determineStatistics(detectedClasses, distances, testY);



%%

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
    [~, ~, ~, ~, ~, TruePositiveRateArray(currentPoint,1), FalsePositiveRateArray(currentPoint,1)] = determineStatistics(detectedClasses, distances, testY);
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
    [~, ~, ~, ~, ~, TruePositiveRateArray(currentPoint,1), FalsePositiveRateArray(currentPoint,1)] = determineStatistics(detectedClasses, distances, validateY);
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
    [~, ~, ~, ~, ~, TruePositiveRateArray(currentPoint,1), FalsePositiveRateArray(currentPoint,1)] = determineStatistics(detectedClasses, distances, testY);
end


[figureHandle] = generateROC(TruePositiveRateArray,FalsePositiveRateArray,"ROC Curve for Test Set");



