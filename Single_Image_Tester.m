clc;
clear;
close all;


%Read in image data to test in network
filepath = 'IMG_4194.jpg';
%filepath = '12146515265_17fefc1709_z.jpg';
%filepath = '14408978343_1c287f68e4_z.jpg';
%filepath = '13512361183_4371843400_z.jpg';

%filepath = '4343697713_9b4298c853_z.jpg';
%filepath = '4340746902_e7c6610e91_z.jpg';
%filepath = '4063353003_4580c5a339_z.jpg';


img = imread(filepath);

%Read in neural network
load('trained_network');

nBlocks = 7;
sampleImageFeatures = normalizeFeatures01(featureExtract(img, nBlocks));

%Classification
[detectedClasses, distances] = predict(net, sampleImageFeatures.');

imshow(img);
if(detectedClasses >=0)
   title("Classification:  Sunset");
else
    title("Classification:  Not A Sunset");
end



%Plotting of gridded figure based on feature extraction methods
%Determine the size of the rows and columns of the image to see if it
%evenly divides into 

[rowsSize, columnsSize, ~] = size(img);

blockRows = floor(rowsSize/nBlocks);
blockColumns = floor(columnsSize/nBlocks);

rows = 1:blockRows:nBlocks * (blockRows+1);
columns = 1:blockColumns:nBlocks * (blockColumns+1);

for index=1:length(rows)
img(rows(index),:,1) = 128;
img(rows(index),:,2) = 128;
img(rows(index),:,3) = 128;
end

for index=1:length(columns)
img(:,columns(index),1) = 128;
img(:,columns(index),2) = 128;
img(:,columns(index),3) = 128;

end

figure
imshow(img);
title("Gridded Image");