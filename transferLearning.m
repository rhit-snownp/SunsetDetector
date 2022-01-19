%Create Datastores for Each Dataset
filepath = "C:\Users\brandejm\Documents\Winter2021\CSSE463\Projects\images\train";
imdsTrain = imageDatastore(filepath,'IncludeSubfolders',true,'LabelSource','foldernames');

filepath = "C:\Users\brandejm\Documents\Winter2021\CSSE463\Projects\images\test";
testingDataStore = imageDatastore(filepath,'IncludeSubfolders',true,'LabelSource','foldernames');

filepath = "C:\Users\brandejm\Documents\Winter2021\CSSE463\Projects\images\validate";
imdsValidation = imageDatastore(filepath,'IncludeSubfolders',true,'LabelSource','foldernames');

numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end


net = alexnet;
%analyzeNetwork(net);
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels))
inputSize = net.Layers(1).InputSize
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2),testingDataStore);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',100, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');
netTransfer = trainNetwork(augimdsTrain,layers,options);
[YPred,scores] = classify(netTransfer,augimdsValidation);

idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

%Get accuracy for each 

%Get training accuracy
[YPredTrain,scores] = classify(netTransfer,augimdsTrain);
Ytrain = imdsTrain.Labels;
trainAccuracy = mean(YPredTrain == Ytrain);

%Get valid accuracy 
[YPredValid,scores] = classify(netTransfer,augimdsValidation);
YValidation = imdsValidation.Labels;
validAccuracy = mean(YPredValid == YValidation);

%Get test accuracy
[YPredTest,scores] = classify(netTransfer,augimdsTest);
YTest = testingDataStore.Labels;
testAccuracy = mean(YPredTest == YTest);


%what am i doing -> 