clc
clear
close all

%Create Feature Vector for Each Type

location = "C:\Users\nicho\OneDrive - Rose-Hulman Institute of Technology\Desktop\Rose-Hulman Schoolwork\Senior Year\Winter\CSSE-463\Projects\Sunset Detector\images\reserved\sunset";
ds = datastore(location);
featureVector = imageDatastoreReader(ds);


currentDirectory = pwd;
cd('../FeatureVectors');
fprintf("Saving Feature Vector");
save("Reserved_Sunset_Feature_Vector",'featureVector');
cd(currentDirectory);