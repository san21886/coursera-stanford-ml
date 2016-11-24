function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%params=[0.01 0.03 0.1 0.3 1 3 10 30];
%errors=[0 0 0];
%for c=1:length(params)
%    C=params(c);
%    for s=1:length(params)
%        sigma=params(s);
%        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
%        predictions = svmPredict(model, Xval);
%        error=[C sigma mean(double(predictions ~= yval))];
%        errors=[errors;error];
%    end
%end
%minimum_error=min(errors(2:end,:)(:,3));
%[i,j]=find(errors(2:end,:)(:,3) == minimum_error);
%C=errors(2:end,:)(i,1)
%sigma=errors(2:end,:)(i,2)

C=1
sigma=0.1





% =========================================================================

end
