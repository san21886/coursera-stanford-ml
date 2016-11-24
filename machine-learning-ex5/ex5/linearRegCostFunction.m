function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

J=sum(((theta' * X')' - y) .^ 2)/(2*m); %cost without regularization.

regularization_term=sum(theta(2:end,:) .^ 2)*lambda/(2*m); %calculate regularization term

J=J+regularization_term;
grad_0=((theta' * X')' - y)' * X(:,1);
grad_0=grad_0/m;
grad_1=((theta' * X')' - y)' * X(:,2:end);
grad_1=(grad_1/m)' + (lambda/m)*theta(2:end,:);
grad=[grad_0;grad_1];









% =========================================================================

grad = grad(:);

end
