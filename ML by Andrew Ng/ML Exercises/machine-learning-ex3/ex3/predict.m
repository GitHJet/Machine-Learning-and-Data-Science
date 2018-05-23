function p = predict(Theta1, Theta2, X)
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

% X is a 5000*400 vector
% Add ones to the X data matrix
X = [ones(m, 1) X]; % X is now a 5000*401 vector

% Theta1 is a 25*401 vector
z2 = Theta1*X'; % z2 is a 25*5000 vector
a2 = sigmoid(z2); 
a2 = a2'; % a2 is now 5000*25

% Theta2 is a 10*26 vector
a2 = [ones(m, 1) a2]; % a2 is now a 5000*26 vector
z3 = Theta2*a2'; % z3 is a 10*5000 vector
a3 = sigmoid(z3); 
a3 = a3'; % a3 is now a 5000*10 vector

% y is a 5000*1 vector
[prob, p] = max(a3, [], 2); % prob is 5000*1 vector

% =========================================================================


end
