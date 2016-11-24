function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

[movie_i,user_j]=find(R==1);
J=0;
for count=1:size(movie_i,1)
    theta_j=Theta(user_j(count),:)';
    x_i=X(movie_i(count),:)';
    diff=(theta_j' * x_i) - Y(movie_i(count),user_j(count));
    diff_square=diff .^ 2;
    J=J+diff_square;
end
J=J/2;

[m,n]=size(X);
for i =1:m
    [movie_i,user_j]=find(R(i,:)==1);
    for k=1:n
        grad_sum=0;
        for count=1:size(user_j,2)
           grad_sum=grad_sum+ (Theta(user_j(count),:) * X(i,:)' - Y(i, user_j(count))) * Theta(user_j(count),k);
        end
        X_grad(i,k)=grad_sum;
    end
end

[m,n]=size(Theta);
for j =1:m
    [movie_i,user_j]=find(R(:,j)==1);
    for k=1:n
        grad_sum=0;
        for count=1:size(movie_i,1)
           grad_sum=grad_sum+ (Theta(j,:) * X(movie_i(count),:)' - Y(movie_i(count), j)) * X(movie_i(count),k);
        end
        Theta_grad(j,k)=grad_sum;
    end
end

J=J+ sum(sum(Theta .^ 2))*lambda/2 + sum(sum(X .^ 2))*lambda/2;
X_grad=X_grad+(lambda*X);
Theta_grad=Theta_grad+(lambda*Theta);



% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
