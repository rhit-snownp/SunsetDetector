function [true_positive, false_positive, true_negative, false_negative, Accuracy, TPR, FPR] = determineCNNStatistics(detectedClasses, distances, yTest)
%Determination of TPR, FPR, TNR, FNR
num_true_positive = 1;
num_false_positive = 1;
num_true_negative = 1;
num_false_negative = 1;
true_positive = 0;
false_positive = 0;
true_negative = 0;
false_negative = 0;

for index=1:length(yTest)
    if(yTest(index) == 'sunset' && detectedClasses(index) == 'sunset')
        true_positive(num_true_positive,1) = index;
        true_positive(num_true_positive,2) = distances(index);
        num_true_positive = num_true_positive + 1;
        
    elseif(yTest(index) == 'sunset' && detectedClasses(index) == 'nonsunset')
        false_negative(num_false_negative,1) = index;
        false_negative(num_false_negative,2) = distances(index);
        num_false_negative = num_false_negative + 1;
      
    elseif(yTest(index) == 'nonsunset' && detectedClasses(index) == 'nonsunset')
        true_negative(num_true_negative,1) = index;
        true_negative(num_true_negative,2) = distances(index);
        num_true_negative = num_true_negative + 1;
        
    elseif(yTest(index) == 'nonsunset' && detectedClasses(index) == 'sunset')
        false_positive(num_false_positive,1) = index;
        false_positive(num_false_positive,2) = distances(index);
        num_false_positive = num_false_positive + 1;
        
        
    else
        fprintf("Error In Classification\n");
    end
        
end

Accuracy = 100 * (num_true_positive + num_true_negative) / (num_true_positive + num_false_positive + num_true_negative + num_false_negative);
TPR = num_true_positive / (num_true_positive + num_false_negative);
FPR = num_false_positive / (num_false_positive + num_true_negative);
% fprintf("The Accuracy is %f, with a TP=%d, a FP=%d, a TN=%d and a FN=%d, TPR=%f, FPR=%f\n",Accuracy,num_true_positive, num_false_positive, num_true_negative, num_false_negative,TPR*100, FPR*100);


end