function F = estimate_fundamental_matrix(matchedPoints1, matchedPoints2)

    x1 = matchedPoints1;
    x2 = matchedPoints2;
    N = size(matchedPoints1,1);
    % Compute the kronecker product
%     A = zeros(N, 9);
%     for i=1:N
%         A(i,:) = kron(x1(i,:), x2(i,:));
%     end
%     
    
    a1 = x1 .* repmat(x2(:,1), [1,3]);
    a2 = x1 .* repmat(x2(:,2), [1,3]);
    A = [a1 a2 x1];

    % Singular Value Decomposition
    [U,S,V] = svd(A);

    % selecting the eigenvector corresponding to the least singular value
    f = V(:,end);
    F = reshape(f, [3,3])';

    % Enforcing Rank=2 

    [U,S,V] = svd(F);
    S(3,3) = 0;     % set last diagonal element to zero
    F = U*S*V'; 
end
