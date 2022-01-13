function getImageFromIndex(store,net,imageIndex,titleString, classification, distance)

    temp = store(2,4);
    testFilenames = [temp{:}];
    correctFilename = string(testFilenames(imageIndex));
    img = imread(correctFilename);

    figure;
    imshow(img);
if(classification >=0)
   title(titleString + " Classification:  Sunset Score: " + abs(distance));
else
   title(titleString + " Classification:  Not A Sunset Score: " + abs(distance));
end


end

