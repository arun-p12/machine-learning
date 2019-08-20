function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% Find Indices of Positive and Negative Examples
pos = find(y==1); neg = find(y == 0);

% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 4);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 4);

% original work
%data = [X, y]			% combine the data
%temp = data(:, 3) == 1		% match y = 1
%pos = data(temp, :)		% get all rows matching y = 1
%temp = data(:, 3) == 0
%neg = data(temp, :)		% get all rows matching y = 0

% plot the graphs against X1 and X2 
% y is differentiated via 'r+' and 'bo'
%plot(pos(:, 1), pos(:, 2), 'r+', 'MarkerSize', 4);
%plot(neg(:, 1), neg(:, 2), 'bo', 'MarkerSize', 4);


% =========================================================================



hold off;

end
