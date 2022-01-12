function [] = vectorContour(x,y,z,maxX,maxY,maxZ)
%this is just doing scatter really, couldnt find a better one 

% x = zeros(100,1);%rand(100,1)*4-2;
% y = zeros(100,1);%rand(100,1)*4-2;
% z = zeros(100,1);%x.*exp(-x.^2-y.^2);
F = TriScatteredInterp(x,y,z);
% ti = -2:.25:2;
% [qx,qy] = meshgrid(ti,ti);
% qz = F(qx,qy);
figure(1);scatter3(x,y,z);
xlabel('kernelScale'); 
ylabel('Bounding Constraint'); 
zlabel('Accuracy on Test Set'); 
title('Hyperparameters vs Accuracy'); 

hold on;
%mesh(qx,qy,qz);
%figure(2);
%contour(qx,qy,qz);
end 