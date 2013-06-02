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
%         variable J. 
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. 
%
%
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% PART 1 

% one pass of feedforward 
a1= [ones(m,1) X];
z2 = Theta1*a1';
a2 = sigmoid(z2);
a2 = [ones(1,m); a2]; % ones for the bias term 
z3 = Theta2*a2;
htheta = sigmoid(z3); 

% Y is y in the matrix form for the given number of classes
Y = zeros(m,num_labels);
Y = y;

Y = Y';
loghtheta = log(htheta);
logoneminus = log(1-htheta);


cost_matrix = -Y.*loghtheta - (1-Y).*logoneminus;
partial_sum = sum(cost_matrix,2);
J = sum(partial_sum)/m; 

% need to skip the bias parameters to compute cost with regularization 

Theta1_nobias = Theta1(:,2:end);
Theta2_nobias = Theta2(:,2:end);
Theta1_nobias = Theta1_nobias.^2; 
Theta2_nobias = Theta2_nobias.^2; 
reg_sum_theta1 = (lambda*sum(sum(Theta1_nobias,2)))/(2*m);
reg_sum_theta2 = (lambda*sum(sum(Theta2_nobias,2)))/(2*m);

J = J + reg_sum_theta1 + reg_sum_theta2;

% PART 2
% to compute the gradient 
delta_output = htheta - Y;

Theta2_nobias = Theta2(:,2:end);

delta_middle = Theta2_nobias'*delta_output.*sigmoidGradient(z2);

%DELTA_outer is between second and third layer
DELTA_outer  = delta_output*a2';
DELTA_inner = delta_middle*a1;

Theta1_grad = (1/m)*DELTA_inner; 
Theta2_grad = (1/m)*DELTA_outer; 

%to account for regularization 
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:, 2:end) + (lambda/m)*Theta2(:,2:end);



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
