% % Example of using a datastore, see 
% 
% clc;
% 
% rootdir = 'C:\Users\brandejm\Documents\Winter2021\CSSE463\Projects\';
% subdir = [rootdir 'featureVectors'];
% 
% trainImages = imageDatastore(...
%     subdir, ...
%     'IncludeSubfolders',true, ...
%     'LabelSource', 'foldernames');
% 
% % Make datastores for the validation and testing sets similarly.
% 
% fprintf('Read images into datastores\n');
% 
% 
% imageDatastoreReader(ds)
currentDirectory = pwd;
cd ('featureVectors');
Files = dir('*.mat');
cd(currentDirectory); 


%Files=dir('FeatureVectors');
store = []; 
for k=1:length(Files)
   FileNames=Files(k).name;
   %newStr = extractAfter(str,12)
   
   if(length(store)==0)
       store = load(FileNames); 
   else
       store = cat(2,store,load(FileNames)); 
   end 
end

%trainData = []; 
first = store(5); 
second = store(6); 
trainData = first.featureVector; 
trainData = cat(1, trainData, second.featureVector); 

trainLabel = first.classificationLabel;
trainLabel = cat(1, trainLabel, second.classificationLabel); 

% labelAndData = cat(2, trainData, trainLabel); 


%have the fitted net, now want to predict on the test data 
first = store(3); 
second = store(4); 
testData = first.featureVector; 
testData = cat(1, testData, second.featureVector); 

testLabel = first.classificationLabel;
testLabel = cat(1, testLabel, second.classificationLabel); 

predictedData = predict(net, testData);

successRate =  sum(abs(predictedData==testLabel))/length(testLabel);



resolution = 20; 
maxKS = 30;
maxBC = 250;
[bestKS, bestBC, bestAccuracy] = hyperparameterGS(resolution,maxKS,maxBC,trainData,trainLabel,testData,testLabel)
net = fitcsvm(trainData, trainLabel, 'KernelFunction', 'rbf', 'KernelScale', bestKS, 'BoxConstraint', bestBC,'Standardize',true); 


% xTrain = imageDatastoreReader(trainImages);
% yTrain = trainImages.Labels;

%% Train and evaluate an SVM







