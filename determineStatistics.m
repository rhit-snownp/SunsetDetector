function [true_positive, false_positive, true_negative, false_negative, Accuracy, TPR, FPR] = determineStatistics(detectedClasses,yTest)
%Determination of TPR, FPR, TNR, FNR
true_positive = 0;
false_positive = 0;
true_negative = 0;
false_negative = 0;


for index=1:length(yTest)
    if(yTest(index) == 1 && detectedClasses(index) == 1)
        true_positive = true_positive + 1;
    elseif(yTest(index) == 1 && detectedClasses(index) == -1)
        false_negative = false_negative + 1;
    elseif(yTest(index) == -1 && detectedClasses(index) == -1)
        true_negative = true_negative + 1;
    elseif(yTest(index) == -1 && detectedClasses(index) == 1)
        false_positive = false_positive + 1;
        
    else
        fprintf("Error In Classification");
    end
        
end

Accuracy = 100 * (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative);
TPR = true_positive / (true_positive + false_negative);
FPR = false_positive / (false_positive + true_negative);
fprintf("The Accuracy is %f, with a TP=%d, a FP=%d, a TN=%d and a FN=%d, TPR=%f, FPR=%f",Accuracy,true_positive, false_positive, true_negative, false_negative,TPR*100, FPR*100);


end