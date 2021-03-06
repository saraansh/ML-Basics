function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

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

hypo = sigmoid(X * theta); %hypothesis
calc = -(y .* log(hypo) + (1-y) .* log(1-hypo));
theta_sq = sum(theta(2:end).^2);
J = (1/m) * sum(calc) + lambda/(2*m) * theta_sq;

% Using Loop
% for i = 1:m,
%     grad = grad + ((hypo(i) - y(i)) * X(i, :)');
% end
% grad = (1/m) * grad + (lambda/m) * [0; theta(2:end)];

% Using Vectorization
grad = (1/m) * sum((hypo - y) .* X)' + (lambda/m) * [0; theta(2:end)];


% =============================================================

end
