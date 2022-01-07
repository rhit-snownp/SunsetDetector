function [feature] = featureExtract(img, nBlocks)


%LST Color Space
%L = R + G + B
%S = R – B
%T = R – 2G + B

%Convert into LST Color Space
Lmatrix = double(img(:,:,1)) + double(img(:,:,2)) + double(img(:,:,3));
Smatrix = double(img(:,:,1)) - double(img(:,:,3));
Tmatrix = double(img(:,:,1)) - 2 * double(img(:,:,2)) + double(img(:,:,3));

%Generate the new double image
LSTImage = zeros(size(img));
LSTImage(:,:,1) = Lmatrix;
LSTImage(:,:,2) = Smatrix;
LSTImage(:,:,3) = Tmatrix;


%Determine the size of the rows and columns of the image to see if it
%evenly divides into 
[rowsSize, columnsSize, ~] = size(LSTImage);

blockRows = round(rowsSize/nBlocks);
blockColumns = round(columnsSize/nBlocks);


n = 1;
feature = zeros(nBlocks*nBlocks*6,1);

for row=1:nBlocks
    for column=1:nBlocks
        startingRow = 1 + (row-1) * blockRows;
        endingRow = (row) * blockRows;
        
        startingColumn = 1 + (column-1) * blockColumns;
        endingColumn = (column) * blockColumns;
        imagePatch(:,:,:) = LSTImage(startingRow:endingRow,startingColumn:endingColumn,:);
        
        L = imagePatch(:,:,1);
        meanL = mean(L,'all');
        stdL = std(L,0,'all');
        feature(n,1) = meanL;
        feature(n+1,1) = stdL;
        
        S = imagePatch(:,:,2);
        meanS = mean(S,'all');
        stdS = std(S,0,'all');
        feature(n+2,1) = meanS;
        feature(n+3,1) = stdS;
        
        T = imagePatch(:,:,3);
        meanT = mean(T,'all');
        stdT = std(T,0,'all');
        feature(n+4,1) = meanT;
        feature(n+5,1) = stdT;
        n = n + 6;
        
        
        
        
    end
end






end

