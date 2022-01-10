clc
clear
close all
tic;
%Create Feature Vector for Each Type


%Reserved Dataset
location = "C:\Users\snownp\OneDrive - Rose-Hulman Institute of Technology\Desktop\Rose-Hulman Schoolwork\Senior Year\Winter\CSSE-463\Projects\Sunset Detector\images\reserved\sunset";
ds = datastore(location);
featureVectorPos = imageDatastoreReader(ds);
classificationLabelPos = ones(length(featureVectorPos),1);

location = "C:\Users\snownp\OneDrive - Rose-Hulman Institute of Technology\Desktop\Rose-Hulman Schoolwork\Senior Year\Winter\CSSE-463\Projects\Sunset Detector\images\reserved\nonsunset";
ds = datastore(location);
featureVectorNeg = imageDatastoreReader(ds);
classificationLabelNeg = -1 * ones(length(featureVectorNeg),1);

featureVector = cat(1,featureVectorPos,featureVectorNeg);
classificationLabel = cat(1,classificationLabelPos,classificationLabelNeg);
currentDirectory = pwd;

cd('FeatureVectors');
fprintf("Saving Feature Vectors\n");
save("Reserved_Feature_Vector",'featureVector','classificationLabel');
cd(currentDirectory);


%Test Dataset
location = "C:\Users\snownp\OneDrive - Rose-Hulman Institute of Technology\Desktop\Rose-Hulman Schoolwork\Senior Year\Winter\CSSE-463\Projects\Sunset Detector\images\test\sunset";
ds = datastore(location);
featureVectorPos = imageDatastoreReader(ds);
classificationLabelPos = ones(length(featureVectorPos),1);

location = "C:\Users\snownp\OneDrive - Rose-Hulman Institute of Technology\Desktop\Rose-Hulman Schoolwork\Senior Year\Winter\CSSE-463\Projects\Sunset Detector\images\test\nonsunset";
ds = datastore(location);
featureVectorNeg = imageDatastoreReader(ds);
classificationLabelNeg = -1 * ones(length(featureVectorNeg),1);

featureVector = cat(1,featureVectorPos,featureVectorNeg);
classificationLabel = cat(1,classificationLabelPos,classificationLabelNeg);
currentDirectory = pwd;

cd('FeatureVectors');
fprintf("Saving Feature Vectors\n");
save("Test_Feature_Vector",'featureVector','classificationLabel');
cd(currentDirectory);

%Train Dataset
location = "C:\Users\snownp\OneDrive - Rose-Hulman Institute of Technology\Desktop\Rose-Hulman Schoolwork\Senior Year\Winter\CSSE-463\Projects\Sunset Detector\images\train\sunset";
ds = datastore(location);
featureVectorPos = imageDatastoreReader(ds);
classificationLabelPos = ones(length(featureVectorPos),1);

location = "C:\Users\snownp\OneDrive - Rose-Hulman Institute of Technology\Desktop\Rose-Hulman Schoolwork\Senior Year\Winter\CSSE-463\Projects\Sunset Detector\images\train\nonsunset";
ds = datastore(location);
featureVectorNeg = imageDatastoreReader(ds);
classificationLabelNeg = -1 * ones(length(featureVectorNeg),1);

featureVector = cat(1,featureVectorPos,featureVectorNeg);
classificationLabel = cat(1,classificationLabelPos,classificationLabelNeg);
currentDirectory = pwd;

cd('FeatureVectors');
fprintf("Saving Feature Vectors\n");
save("Train_Feature_Vector",'featureVector','classificationLabel');
cd(currentDirectory);

%Validation Dataset
location = "C:\Users\snownp\OneDrive - Rose-Hulman Institute of Technology\Desktop\Rose-Hulman Schoolwork\Senior Year\Winter\CSSE-463\Projects\Sunset Detector\images\validate\sunset";
ds = datastore(location);
featureVectorPos = imageDatastoreReader(ds);
classificationLabelPos = ones(length(featureVectorPos),1);

location = "C:\Users\snownp\OneDrive - Rose-Hulman Institute of Technology\Desktop\Rose-Hulman Schoolwork\Senior Year\Winter\CSSE-463\Projects\Sunset Detector\images\validate\nonsunset";
ds = datastore(location);
featureVectorNeg = imageDatastoreReader(ds);
classificationLabelNeg = -1 * ones(length(featureVectorNeg),1);

featureVector = cat(1,featureVectorPos,featureVectorNeg);
classificationLabel = cat(1,classificationLabelPos,classificationLabelNeg);
currentDirectory = pwd;

cd('FeatureVectors');
fprintf("Saving Feature Vectors\n");
save("Validation_Feature_Vector",'featureVector','classificationLabel');
cd(currentDirectory);





elapsedTime = toc;

fprintf("All Images Processed Into Feature Vectors In %f Seconds\n",elapsedTime);