function X1 = algebraicTriangulation(x1, x2, P1, P2)
    X1 = zeros(4,size(x1,2));
    for i = 1:size(x1, 2)
        a1 = x1(2,i)*P1(3,:) - P1(2,:);
        a2 = x1(1,i)*P1(3,:) - P1(1,:);
        a3 = x2(2,i)*P2(3,:) - P2(2,:);
        a4 = x2(1,i)*P2(3,:) - P2(1,:);
        A = [a1; a2; a3; a4];
        [U, S, V] = svd(A);
        X1(:,i) = V(:,end);
        X1(:,i) = X1(:,i)./X1(4,i);
    end
end

