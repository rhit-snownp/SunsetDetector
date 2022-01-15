function [bestKS, bestBC, bestAccuracy,meshKS,meshBC,meshAcc,numSupportVectors] = optimalGridSearch(resolution,minKS,maxKS,minBC,maxBC,xTrain,yTrain,xTest,yTest)



bestKS = 0;
bestBC = 0;
bestAccuracy = 0;


meshKS = [];
meshBC = [];
meshAcc = [];
numSupportVectors = [];

ROCstuff = zeros(resolution*resolution,2);
dataAndConstraints = zeros(resolution*resolution,3); 

for kS = 1:resolution
    for bC = 1:resolution 
        index = resolution*kS+bC-resolution; 
        
        kernelScale = kS*(maxKS-minKS)/resolution+minKS;
        boxConstraint = bC*(maxBC-minBC)/resolution+minBC;
        
        meshKS = cat(1,kernelScale,meshKS); 
        meshBC = cat(1,boxConstraint,meshBC);

        dataAndConstraints(index,1) = kernelScale; 
        dataAndConstraints(index,2) = boxConstraint;
        
        net = fitcsvm(xTrain, yTrain, 'KernelFunction', 'rbf', 'KernelScale', kernelScale, 'BoxConstraint', boxConstraint); 
        numSupportVectors = cat(1, net.SupportVectors, numSupportVectors); 
        predictedData = predict(net, xTest);
        
        successRate =  sum(abs(predictedData==yTest))/length(yTest);
        meshAcc = cat(1,successRate,meshAcc); 
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

