function [feature] = featureExtract(img, nBlocks)


%LST Color Space
%L = R + G + B
%S = R – B
%T = R – 2G + B

%Convert into LST Color Space
L = double(img(:,:,1)) + double(img(:,:,2)) + double(img(:,:,3));
S = double(img(:,:,1)) - double(img(:,:,3));
T = double(img(:,:,1)) - 2 * double(img(:,:,2)) + double(img(:,:,3));

%Generate the new double image
LSTImage = zeros(size(img));
LSTImage(:,:,1) = L;
LSTImage(:,:,2) = S;
LSTImage(:,:,3) = T;


%Determine the size of the rows and columns of the image to see if it
%evenly divides into 
[rowsSize, columnsSize, ~] = size(LSTImage);

blockRows = round(rowsSize/nBlocks);
blockColumns = round(columnsSize/nBlocks);

1,7,14,21,28
1,2,3,4,5,

n = 1;
feature = zeros(1,294);

for row=1:nBlocks
    for column=1:nBlocks
        startingRow = 1 + (row-1) * blockRows;
        endingRow = 1 + (row) * blockRows;
        
        startingColumn = 1 + (column-1) * blockColumns;
        endingColumn = 1 + (column) * blockColumns;
        imagePatch(:,:,:) = LSTImage(startingRow:endingRow,startingColumn:endingColumn,:);
        
        L = imagePatch(:,:,1);
        meanL = mean(L);
        stdL = std(L(:));
        feature(n,1) = meanL;
        feature(n+1,1) = stdL;
        
        S = imagePatch(:,:,2);
        meanS = mean(S);
        stdS = std(S(:));
        feature(n+2,1) = meanS;
        feature(n+3,1) = stdS;
        
        T = imagePatch(:,:,2);
        meanT = mean(T);
        stdT = std(T(:));
        feature(n+4,1) = meanT;
        feature(n+5,1) = stdT;
        n = n + 6;
        
        
        
        
    end
end






end

