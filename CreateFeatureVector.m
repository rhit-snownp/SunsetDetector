clc
clear
close all
tic;
%Create Feature Vector for Each Type


%Reserved Dataset
location = "C:\Users\snownp\OneDrive - Rose-Hulman Institute of Technology\Desktop\Rose-Hulman Schoolwork\Senior Year\Winter\CSSE-463\Projects\Sunset Detector\images\reserved\sunset";
ds = datastore(location);
featureVector = imageDatastoreReader(ds);
classificationLabel = ones(length(featureVector),1);

currentDirectory = pwd;
cd('FeatureVectors');
fprintf("Saving Feature Vector");
save("Reserved_Sunset_Feature_Vector",'featureVector','classificationLabel');
cd(currentDirectory);

location = "C:\Users\snownp\OneDrive - Rose-Hulman Institute of Technology\Desktop\Rose-Hulman Schoolwork\Senior Year\Winter\CSSE-463\Projects\Sunset Detector\images\reserved\nonsunset";
ds = datastore(location);
featureVector = imageDatastoreReader(ds);
classificationLabel = -1 * ones(length(featureVector),1);

currentDirectory = pwd;
cd('FeatureVectors');
fprintf("Saving Feature Vector");
save("Reserved_NonSunset_Feature_Vector",'featureVector','classificationLabel');
cd(currentDirectory);


%Test Dataset
location = "C:\Users\snownp\OneDrive - Rose-Hulman Institute of Technology\Desktop\Rose-Hulman Schoolwork\Senior Year\Winter\CSSE-463\Projects\Sunset Detector\images\test\sunset";
ds = datastore(location);
featureVector = imageDatastoreReader(ds);
classificationLabel = ones(length(featureVector),1);

currentDirectory = pwd;
cd('FeatureVectors');
fprintf("Saving Feature Vector");
save("Test_Sunset_Feature_Vector",'featureVector','classificationLabel');
cd(currentDirectory);

location = "C:\Users\snownp\OneDrive - Rose-Hulman Institute of Technology\Desktop\Rose-Hulman Schoolwork\Senior Year\Winter\CSSE-463\Projects\Sunset Detector\images\test\nonsunset";
ds = datastore(location);
featureVector = imageDatastoreReader(ds);
classificationLabel = -1 * ones(length(featureVector),1);

currentDirectory = pwd;
cd('FeatureVectors');
fprintf("Saving Feature Vector");
save("Test_NonSunset_Feature_Vector",'featureVector','classificationLabel');
cd(currentDirectory);



%Training Dataset
location = "C:\Users\snownp\OneDrive - Rose-Hulman Institute of Technology\Desktop\Rose-Hulman Schoolwork\Senior Year\Winter\CSSE-463\Projects\Sunset Detector\images\train\sunset";
ds = datastore(location);
featureVector = imageDatastoreReader(ds);
classificationLabel = ones(length(featureVector),1);

currentDirectory = pwd;
cd('FeatureVectors');
fprintf("Saving Feature Vector");
save("Train_Sunset_Feature_Vector",'featureVector','classificationLabel');
cd(currentDirectory);

location = "C:\Users\snownp\OneDrive - Rose-Hulman Institute of Technology\Desktop\Rose-Hulman Schoolwork\Senior Year\Winter\CSSE-463\Projects\Sunset Detector\images\train\nonsunset";
ds = datastore(location);
featureVector = imageDatastoreReader(ds);
classificationLabel = -1 * ones(length(featureVector),1);

currentDirectory = pwd;
cd('FeatureVectors');
fprintf("Saving Feature Vector");
save("Train_NonSunset_Feature_Vector",'featureVector','classificationLabel');
cd(currentDirectory);



%Validation Dataset
location = "C:\Users\snownp\OneDrive - Rose-Hulman Institute of Technology\Desktop\Rose-Hulman Schoolwork\Senior Year\Winter\CSSE-463\Projects\Sunset Detector\images\validate\sunset";
ds = datastore(location);
featureVector = imageDatastoreReader(ds);
classificationLabel = ones(length(featureVector),1);

currentDirectory = pwd;
cd('FeatureVectors');
fprintf("Saving Feature Vector");
save("Validation_Sunset_Feature_Vector",'featureVector','classificationLabel');
cd(currentDirectory);

location = "C:\Users\snownp\OneDrive - Rose-Hulman Institute of Technology\Desktop\Rose-Hulman Schoolwork\Senior Year\Winter\CSSE-463\Projects\Sunset Detector\images\validate\nonsunset";
ds = datastore(location);
featureVector = imageDatastoreReader(ds);
classificationLabel = -1 * ones(length(featureVector),1);

currentDirectory = pwd;
cd('FeatureVectors');
fprintf("Saving Feature Vector\n");
save("Validation_NonSunset_Feature_Vector",'featureVector','classificationLabel');
cd(currentDirectory);


elapsedTime = toc;

fprintf("All Images Processed Into Feature Vectors In %f Seconds\n",elapsedTime);