function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% BRL - Linear regression, so calculate 'ha
h=X*Theta';
% Then feed in to cost function.  As this is Collaborative Filtering,
% multiply h-Y by R matrix (containing 1 where user reviewed film, 
% 0 otherwise.  This will exclude all obs where no user/film review
% exists by multiplying by 0.
J=(1/2)*sum(sum(((h-Y).*R).^2));

% Regularized
J=(1/2)*sum(sum(((h-Y).*R).^2))... 
		+((lambda/2)*sum(sum(Theta.^2)))...
		+((lambda/2)*sum(sum(X.^2)));

% For test case...
% 4 Users
% 5 Movies
% 3 Features

% X = 5x3 (movies x features)
% Y = 5x4 (movies x users)
% R = 5x4 (movies x users)
% Theta 4x3 (users x features)
% h 5x4 (movies x users)

% X_grad = 5x3 
% Theta_grad = 4x3

% First, filter out entries from user/movie matrix (Y)
% where there is no review, along with corresponding
% entries in predictions (h (X*Theta')) 
YR=Y.*R;
hR=h.*R;

% Now we have 'filtered' data, we can more easily
% calculate gradients for X and Theta as follows..

% X_grad
for i = 1:size(X_grad,1)
	% Non-regularized
	%X_grad(i,:)=(hR(i,:)-YR(i,:))*Theta;
	%Regularized - using product of lamdba x features of current movie	
	X_grad(i,:)=((hR(i,:)-YR(i,:))*Theta) + (lambda*X(i,:));
end

% Theta_grad
for i = 1:size(Theta_grad,1)
	%Non-regularized
	%Theta_grad(i,:)=(hR(:,i)-YR(:,i))'*X;
	%Regularized - using product of lambda * feature weights of current user
	Theta_grad(i,:)=((hR(:,i)-YR(:,i))'*X) + (lambda*Theta(i,:));
end


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
