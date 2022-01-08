function features = imageDatastoreReader(ds)
% Example of using an image datastore.

nBlocks = 7; % 
nImages = numel(ds.Files);

features = zeros(nImages, nBlocks * nBlocks * 6); 
row = 1;
for i = 1:nImages
    [img, fileinfo] = readimage(ds, i);
    % fileinfo struct with filename and another field.
    fprintf('Processing %s\n', fileinfo.Filename);
    featureVector = featureExtract(img, nBlocks);
    features(row,:) = featureVector;
    row = row + 1;
end
