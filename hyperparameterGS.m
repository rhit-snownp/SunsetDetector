function [bestKS, bestBC, bestAccuracy,meshKS,meshBC,meshAcc,numSupportVectors] = hyperparameterGS(resolution,maxKS,maxBC,xTrain,yTrain,xTest,yTest)

%iterate this 3 times for zooming in with resolution of 10 


%resolution = 10; 
minKS = 0; 
minBC = 0; 
bestKS = 0;
bestBC = 0;
bestAccuracy = 0;
numIterations = 3;

ROCstuff = zeros(resolution*resolution,2);
dataAndConstraints = zeros(resolution*resolution,3);
meshKS = zeros(1,numIterations*resolution*resolution); 
meshBC = zeros(1,numIterations*resolution*resolution); 
meshAcc = zeros(1,numIterations*resolution*resolution); 
numSupportVectors = zeros(1,numIterations*resolution*resolution); 

index2 = 0; 

for numIterations = 1:3 
    for kS = 1:resolution
        for bC = 1:resolution 
            index2 = index2+1;
            index = resolution*kS+bC-resolution; 

            kernelScale = kS*(maxKS-minKS)/resolution+minKS;
            boxConstraint = bC*(maxBC-minBC)/resolution+minBC;

            meshKS(1,index2) = kernelScale;  
            meshBC(1,index2) = boxConstraint;
            
            dataAndConstraints(index,1) = kernelScale; 
            dataAndConstraints(index,2) = boxConstraint;

            net = fitcsvm(xTrain, yTrain, 'KernelFunction', 'rbf', 'KernelScale', kernelScale, 'BoxConstraint', boxConstraint); 
            
            numSupportVectors(1,index2) = size(net.SupportVectors,1); 
            
            predictedData = predict(net, xTest);
            
%             disp(size(predictedData))
%             disp(size(yTest))
            
            successRate =  sum(abs(predictedData==yTest))/length(yTest);
            meshAcc(1,index2) = successRate; 
            dataAndConstraints(index,3) = successRate; 
            TPrate = 100*length(find(predictedData == 1 & yTest == 1))/length(find(yTest==1));
            FPrate = 100*length(find(predictedData == 1 & yTest == -1))/length(find(yTest==-1));

            ROCstuff(index,2) = FPrate;
            ROCstuff(index,1) = TPrate; 


        end 
    end

    %Printing optimal Settings and Rate 
    [M,I] = max(dataAndConstraints(:,3));
    bestKS = dataAndConstraints(I,1);
    bestBC = dataAndConstraints(I,2);
    bestAccuracy = M;

    minKS = bestKS-(maxKS-minKS)/resolution;
    minKS = bestKS+(maxKS-minKS)/resolution;

    minBC = bestBC-(maxBC-minBC)/resolution;
    minBC = bestBC+(maxBC-minBC)/resolution;

end 

vectorContour(meshKS,meshBC,meshAcc); 
disp(max(meshAcc)); 





% disp("Best Accuracy: "+bestAccuracy)
% disp("Best KS: "+bestKS)
% disp("Best BC: "+bestBC)
% 
% scatter(ROCstuff(:,2)/100,ROCstuff(:,1)/100)
% xlabel('False Positive Rate'); 
% ylabel('True Positive Rate'); 
% title('ROC Curve'); 