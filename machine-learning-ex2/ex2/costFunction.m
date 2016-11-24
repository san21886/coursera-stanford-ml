function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Evaluating J i.e cost for given theta.
for i = 1:m
    label=y(i);
    ith_training_ex=X(i,:);
    z=theta' *  ith_training_ex';
    sigmod_of_z=sigmoid(z);
    log_of_pos_sigmoid=log(sigmod_of_z);
    log_of_neg_sigmoid=log(1 - sigmod_of_z);
    J = J + (-label)*log_of_pos_sigmoid - (1-label)*log_of_neg_sigmoid;
end
J=J/m;

for j = 1:length(theta)
    sum=0;
    for i = 1:m
        label=y(i);
        ith_training_ex=X(i,:);
        z=theta' *  ith_training_ex';
        sigmod_of_z=sigmoid(z);
        sum = sum + (sigmod_of_z - label) * ith_training_ex(j);
    end
    grad(j)=sum/m;
end


% Evaluating gradient.




% =============================================================

end
