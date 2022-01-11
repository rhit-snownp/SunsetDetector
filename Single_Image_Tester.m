clc;
clear all;
close all;


%Read in image data to test in network
%filepath = '14408978343_1c287f68e4_z.jpg';
%filepath = '13512361183_4371843400_z.jpg';

%filepath = '4343697713_9b4298c853_z.jpg';
%filepath = '4340746902_e7c6610e91_z.jpg';
filepath = '4063353003_4580c5a339_z.jpg';


img = imread(filepath);

%Read in neural network
load('trained_network');

nBlocks = 7;
sampleImageFeatures = featureExtract(img, nBlocks);

%Classification
[detectedClasses, distances] = predict(net, sampleImageFeatures.');

imshow(img);
if(detectedClasses >=0)
   title("Classification:  Sunset");
else
    title("Classification:  Not A Sunset");
end