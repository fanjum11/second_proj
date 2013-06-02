%% Machine Learning Online Class - Exercise 4 Neural Network Learning


%% Initialization
clear ; close all; clc
                       

%% =========== Part 1: Loading Data =============
%  We start the exercise by first loading the dataset. 
%

% Load Training Data
% y is a vector of 0 or 1; 0 for dead and 1 for alive

data_all = load('train1_octave_format.csv');
%data = data(1:20,:);

TOTAL_rows = size(data_all,1);
START_ROW = 100;
ROW_INCR = 100; 
Count = 0;
pred_val = [];
pred_train = []; 

for data_rows = START_ROW:ROW_INCR:TOTAL_rows  % limit rows
%X = data(:, [1, 2]); y = data(:, 3); 
dataval = data_all(START_ROW+1:TOTAL_rows,:);
data = data_all(1:START_ROW,:);
s_data = size(data,2);
X = data(:,2:s_data); y = data(:, 1); 
m = size(X, 1);
Xval = dataval(:,2:s_data);
yval = dataval(:,1);

Count = Count + 1 
%NN parameters
input_layer_size  = size(X,2);  % equals number of columns in X with each column as a feature 
hidden_layer_size = 25;   % 25 hidden units
num_labels = 1;          % 1 label   


%% ================ Part 6: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];



%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 2000);

%  You should also try different values of lambda
lambda = 3;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred_t = predict(Theta1,Theta2,X);
pred_v = predict(Theta1, Theta2, Xval);

%fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == yval)) * 100);

pred_train(Count) = mean(double(pred_t== y))*100;
pred_val(Count) = 100-mean(double(pred_v== yval))*100;

end % for data_rows = START_ROW:ROW_INCR:TOTAL_rows  

pred_train
pred_val

% REAL data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%READ IN THE REAL FILE
X = [];
data = [];
data = load('test_octave_format.csv');
s_data = size(data,2);
X = data(:,1:s_data); 

[m, n] = size(X);

% Add intercept term to x and X_test
%X = [ones(m, 1) X];

% Compute accuracy on our training set
p = predict(Theta1, Theta2, X);
round(p);
save -ascii results1.csv p


