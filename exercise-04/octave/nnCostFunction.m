function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% compute the hypothesis
X1 = [ones(size(X)(1), 1) X];     % prefix X0
ax = sigmoid(X1 * Theta1');

% repeat the two steps if there are additional hidden layers
ax = [ones(size(ax)(1), 1) ax];  % prefix a0
ax = sigmoid(ax * Theta2');

% now we're at the final / output layer
% convert y to a collection of vectors... in the oneVsAll format
% each vector has a 1 for its classification, and a 0 for all else
y_all = zeros(m, num_labels);
for i = 1:num_labels
   pos = find(y == i);
   y_all(pos, i) = 1;
end

% calculate the cost function, by retaining the classified value, and
% discard all the rest.... element-wise multiplication by 1s and 0s will ensure that
J = sum(sum((-y_all .* log(ax)) - ((1 - y_all) .* log(1 - ax)))) / m;

% compute the regularization
Theta1_Reg = Theta1;
Theta2_Reg = Theta2;
Theta1_Reg(:, 1) = 0;          % null out the zeroth column
Theta2_Reg(:, 1) = 0;          % null out the zeroth column
reg = (lambda/(2*m)) * (sum(sum(Theta1_Reg .* Theta1_Reg)) + sum(sum(Theta2_Reg .* Theta2_Reg)));

% update the cost function
J = J + reg;


% back propagation
% cost gradient  ... using forward propagation
for t = 1:m
    % step #1 :: perform forward propagation, stepping thru each layer
    a1 = X1(t, :);                  % X1 already has X0 (the ones column) added

    z2 = a1 * Theta1';
    a2 = sigmoid(z2);

    z3 = [1 a2] * Theta2';          % add in the zeroth element for a2
    a3 = sigmoid(z3);

    % step #2 :: calculate the resultant error
    % δ(3) = (a(3) − yk)
    ytemp = y(t, :);                % save the label
    y3 = zeros(1, num_labels);      % create a column of labels for oneVsAll
    y3(1, ytemp) = 1;               % update the appropriate column based on the label

    delta_3 = a3 - y3;              % compute the error at the final layer
    delta_3 = delta_3';             % transform into a vector

    % step #3 :: propagate back, and calculate the gradient error
    % δ(2) =  Θ(2) T δ(3). ∗ g′(z(2))
    % don't have to worry about delta_1 since that's the input layer... no error
    % transform the output of the sigmoidGradient into a column vector
    % Add the bias node to z2, while calculating the gradient.
    delta_2 = (Theta2' * delta_3) .* sigmoidGradient([1  z2])';
    % strip out the delta for the 0th unit... we had added that unit in
    delta_2 = delta_2(2:end, 1);

    % step #4 :: accumulate the gradient
    % ∆(l) = ∆(l) + δ(l+1)(a(l))T
    Theta2_grad = Theta2_grad + (delta_3 * [1 a2]);    % add in the bias for a2
    Theta1_grad = Theta1_grad + (delta_2 * a1);        % a1 is the input layer... no bias
end

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% Regularized gradient
% add the term (lambda / m) .* Theta(layer) to each term
% no need to regularize for the first layer
Theta1_grad = Theta1_grad + (lambda / m) .* Theta1_Reg;
Theta2_grad = Theta2_grad + (lambda / m) .* Theta2_Reg;

% alternate implementation
%Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ( (lambda/m) * Theta1(:, 2:end) );
%Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ( (lambda/m) * Theta2(:, 2:end) );







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
