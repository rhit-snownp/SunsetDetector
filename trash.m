function [] = vectorContour(x,y,z)

% x = zeros(100,1);%rand(100,1)*4-2;
% y = zeros(100,1);%rand(100,1)*4-2;
% z = zeros(100,1);%x.*exp(-x.^2-y.^2);
F = TriScatteredInterp(x,y,z);
ti = -2:.25:2;
[qx,qy] = meshgrid(ti,ti);
qz = F(qx,qy);
figure(1);scatter3(x,y,z);
hold on;
mesh(qx,qy,qz);
figure(2);
contour(qx,qy,qz);
end 