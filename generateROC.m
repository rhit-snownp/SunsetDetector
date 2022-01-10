function [figureHandle] = generateROC(TruePositiveRateArray,FalsePositiveRateArray,figureTitle)
% Plot an ROC curve for an image. 


% Create a new figure. You can also number it: figure(1)
figureHandle = figure;
% Hold on means all subsequent plot data will be overlaid on a single plot
hold on;
% Plots using a blue line (see 'help plot' for shape and color codes 
plot(FalsePositiveRateArray, TruePositiveRateArray, 'b-', 'LineWidth', 2);
% Overlaid with circles at the data points
plot(FalsePositiveRateArray, TruePositiveRateArray, 'bo', 'MarkerSize', 1, 'LineWidth', 1);


% Title, labels, range for axes
title(figureTitle, 'fontSize', 18); % Really. Change this title.
xlabel('False Positive Rate', 'fontWeight', 'bold');
ylabel('True Positive Rate', 'fontWeight', 'bold');
axis([0 1 0 1]);
end



