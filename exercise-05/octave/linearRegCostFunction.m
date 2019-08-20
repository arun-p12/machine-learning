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

hx = X * theta ;         %  m x n  * n x 1   = m x 1
diff = hx - y;
sse = diff .^ 2;	 % square of the errors
ov2m = 1 / ( 2*m);
J = sum(sse) * ov2m;     %  the cost function

theta(1) = 0;
reg = (lambda * ov2m) * sum(theta .* theta);
J = J + reg;

ovm = 2 * ov2m;
grad = (X' * diff) * ovm;
grad = grad + (lambda * ovm * theta);








% =========================================================================

grad = grad(:);

end
