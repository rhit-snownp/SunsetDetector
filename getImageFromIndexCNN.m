function getImageFromIndex(store,net,imageIndex,titleString, classification, distance)

    correctFilename = string(store.Files(imageIndex,1));
    img = imread(correctFilename);

    figure;
    imshow(img);
if(classification >=0)
   title(titleString + " Classification:  Sunset Score: " + abs(distance));
else
   title(titleString + " Classification:  Not A Sunset Score: " + abs(distance));
end


end

