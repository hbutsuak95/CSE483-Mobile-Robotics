function [newpts, T] = normalize2DPoints(pts)

    % Convert to Homogeneous coordinates
   pts(:,3) = 1;
    mu = mean(pts(:,1:2));          
    newp(:,1) = pts(:,1)-mu(1); 
    newp(:,2) = pts(:,2)-mu(2);
    
    dist = sqrt(newp(:,1).^2 + newp(:,2).^2);
    meandist = mean(dist(:));  
    
    scale = sqrt(2)/meandist;
    
    T = [scale   0   -scale*mu(1)
         0     scale -scale*mu(2)
         0       0      1      ];
    newpts = T*pts';
end