%working on generating relevant ROC curve 
%our own optimization
%Get each type of success bestKS bestBC
%Kernel Scale: 26.712044, Box Constraint: 188.352500
% bestBC = 106.000000 ;
% bestKS = 12.720000;
net = fitcsvm(trainX, trainY, 'KernelFunction', 'rbf', 'KernelScale', 12.720000 , 'BoxConstraint',106.000000); 
%now, trying to find num support vectors
numSV = net.SupportVectors;
[trainPred, distances] = predict(net, trainX);
successRate =  sum(abs(trainPred==trainY))/length(trainY);

[validPred, distances] = predict(net, validateX);
successRate =  sum(abs(validPred==validateY))/length(validateY);

[testPred, distances] = predict(net, testX);
successRate =  sum(abs(testPred==testY))/length(testY);
% 
% [reservedPred, distances] = predict(net, reservedX);
% successRate =  sum(abs(reservedPred==reservedY))/length(reservedY);

[true_positive, false_positive, true_negative, false_negative, Accuracy, TPR, FPR] = determineStatistics(testPred, distances, testY)


