function pred = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


X = [ones(m, 1) X];
prob = sigmoid(X * Theta1');
t = size(prob, 1);
prob = [ones(t, 1) prob];
prob = sigmoid(prob * Theta2');

% initialize our prediction vector to 0
pred = zeros(m, 1);

% step thru each digit in the training set, and get the column with highest probability
% the column number corresponds to the digit ... with 0 as column 10
for i = 1:m
   [val, col] = max(max(prob(i, :), [], 1));         % the last arg = 2 for row
   if(col == 10)
      col = 0;
   end
   pred(i) = col;
end







% =========================================================================


end
