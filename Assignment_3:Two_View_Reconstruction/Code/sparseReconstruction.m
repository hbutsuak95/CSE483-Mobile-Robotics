K =  [  558.7087    0.0000  310.3210 ;
     0.0000  558.2827  240.2395 ;
     0.0000    0.0000    1.0000 ] ;

img1 = imread('../images/img1.png');
img2 = imread('../images/img2.png');

img1 = rgb2gray(img1);
img2 = rgb2gray(img2);

interest_points1 = detectSURFFeatures(img1,'MetricThreshold',10);
interest_points2 = detectSURFFeatures(img2,'MetricThreshold',10);

[features1,validPoints1] = extractFeatures(img1,interest_points1);
[features2,validPoints2] = extractFeatures(img2,interest_points2);

indexPairs = matchFeatures(features1,features2);

matchedPoints1 = validPoints1(indexPairs(:,1),:);
matchedPoints2 = validPoints2(indexPairs(:,2),:);

[norm_matchedPoints1, T1] = normalize2DPoints(matchedPoints1.Location);
[norm_matchedPoints2, T2] = normalize2DPoints(matchedPoints2.Location);

% norm_matchedPoints1 = norm_matchedPoints1(1:2,:)';
% norm_matchedPoints2 = norm_matchedPoints2(1:2,:)';


[F, inliers] = estimateFundamentalMatrixRANSAC(norm_matchedPoints1', norm_matchedPoints2', 10000);

% 
F = T2'*F*T1;

% Retrieve Essential Matrix from the Fundamental Matrix

E = K'*F*K;

% setting 
[U,S,V] = svd(E);
l = (S(1,1)+S(2,2))/2.0;
S(1,1) = l;
S(2,2) = l;
S(3,3) = 0;
E = U*S*V';

points1 = matchedPoints1.Location;
points2 = matchedPoints2.Location;
points1(:,3) = 1;
points2(:,3) = 1;

inlier_points1 = points1(inliers,:);
inlier_points2 = points2(inliers,:);

%[R, t] = decomposeEssentialMatrix(E, points1', points2', K)
[R, t] = decomposeEssentialMatrix(E, inlier_points1', inlier_points2', K);

P1 = K*[eye(3) [0 0 0]'];
P2 = K*[R t];
%pts_3D = algebraicTriangulation(points1', points2', P1, P2)';
pts_3D = algebraicTriangulation(inlier_points1', inlier_points2', P1, P2)';


figure
hold on;
grid on;
xlim([-2, 6])
ylim([-2, 6])
zlim([-2, 6])

xlabel("X axis")
ylabel("Y axis")
zlabel("Z axis")
colormap(parula(10))

c = pts_3D(:,3);
c(find(c>6)) = 6;

%c = (pts_3D(:,3)-min(pts_3D(:,3)))./(max(pts_3D(:,3)) - min(pts_3D(:,3)))*100

scatter3(pts_3D(:,1), pts_3D(:,2), pts_3D(:,3),40,c,'filled')
hold on
plotCameraFrustum(eye(4), 'b', .1)
hold on
plotCameraFrustum([R t; [0 0 0 1]], 'r', .1)


F

R

t


figure
showMatchedFeatures(img1,img2,matchedPoints1(1:100,:),matchedPoints2(1:100,:),'montage','PlotOptions',{'ro','go','y--'});

legend('matched points 1','matched points 2');