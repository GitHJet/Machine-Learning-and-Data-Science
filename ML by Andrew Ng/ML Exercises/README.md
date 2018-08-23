### Learning Points from Assignments

## Week 1

Learning Points:
1. Linear regression cost function implementation
2. Feature normalization implementation
3. Gradient descent step implementation
4. Selecting learning rates
5. Normal equation

**Linear regression cost function, J**

`J = 1/2m * sum(sqrErrors), where sqrErrors = (X*theta - y).^2;`

**Feature normalization of X**

`X = (X .- mean(X)) ./ std(X);`

**Gradient descent single step**

`theta = theta - alpha * partial_derivation(J) = theta - alpha * 1/m * X' * (X*theta - y); (Vectorized)`

**Learning rate**

Gradient descent converges slowly if learning rate is small, and does not converge/diverge if learning rate is too big.

**Normal equation**

`theta = (pinv(X'*X)) * (X'*y);`

Normal equation exactly calculates theta and does not require feature normalization. Slow if n is too larg, eg. n > 10,000.


## Week 2

Learning Points:
1. Sigmoid function
2. Logreg cost function and gradient
3. Regularized logreg cost function
4. Regularization and overfitting

**Sigmoid function**

`sigmoid(z) = 1 ./ (1 + e.^(-z));`

**Logreg cost function and gradient**
```
h = sigmoid(X*theta);
J = 1/m * (-y' * log(h) - (1-y)' * log(1-h));
grad = 1/m * X' * (h - y);
```
**Regularized logreg cost function**
```
h = sigmoid(X*theta);
J = 1/m * (-y' * log(h) - (1-y)' * log(1-h));
theta(1) = 0; % This prevents regularization of theta0 (Matlab indexing starts at 1)
J = J + lambda/(2*m) * sum(theta'*theta);
grad = 1/m * X' * (h - y) + lambda/m * theta; %
```
**Regularization and overfitting**

No regularization (lambda=0) causes overfitting while too much regularization (lambda=100) causes underfitting.


## Week 3

Learning Points:
1. Vectorization makes code faster than looping
2. One-vs-all/Multi-class classification
3. Logistic regression vs neural networks
4. Feedforward propagation and prediction

**One-vs-all/Multi-class classification**

Train each classifier independently using a for-loop from 1 to K.

**Logistic regression vs neural networks**

Logistic regression cannot form complex hypotheses like neural networks because it is a linear classifier.

**Feedforward propagation and prediction**
```
% X is a vector of inputs.
% Add ones to the X data matrix
X = [ones(m, 1) X];

% Theta1 is a vector of theta to map input layer to hidden layer.
z2 = Theta1*X'; 
a2 = sigmoid(z2); 
a2 = a2';

% Theta2 is a vector of theta to map hidden layer to final layer.
a2 = [ones(m, 1) a2];
z3 = Theta2*a2';
a3 = sigmoid(z3); 
a3 = a3';

% p is the prediction, the choice with the highest probability.
[prob, p] = max(a3, [], 2);
```





**Derivation of sigmoid function gradient, expressed as a function of the function value** (only σ(x), but not x, is present)

<a href="https://www.codecogs.com/eqnedit.php?latex=\LARGE&space;\begin{align*}\sigma(x)'&=\left(\frac{1}{1&plus;e^{-x}}\right)'=\frac{-(1&plus;e^{-x})'}{(1&plus;e^{-x})^2}=\frac{-1'-(e^{-x})'}{(1&plus;e^{-x})^2}=\frac{0-(-x)'(e^{-x})}{(1&plus;e^{-x})^2}=\frac{-(-1)(e^{-x})}{(1&plus;e^{-x})^2}=\frac{e^{-x}}{(1&plus;e^{-x})^2}&space;\newline&space;&=\left(\frac{1}{1&plus;e^{-x}}\right)\left(\frac{e^{-x}}{1&plus;e^{-x}}\right)=\sigma(x)\left(\frac{&plus;1-1&space;&plus;&space;e^{-x}}{1&plus;e^{-x}}\right)=\sigma(x)\left(\frac{1&space;&plus;&space;e^{-x}}{1&plus;e^{-x}}&space;-&space;\frac{1}{1&plus;e^{-x}}\right)=\sigma(x)(1&space;-&space;\sigma(x))\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\LARGE&space;\begin{align*}\sigma(x)'&=\left(\frac{1}{1&plus;e^{-x}}\right)'=\frac{-(1&plus;e^{-x})'}{(1&plus;e^{-x})^2}=\frac{-1'-(e^{-x})'}{(1&plus;e^{-x})^2}=\frac{0-(-x)'(e^{-x})}{(1&plus;e^{-x})^2}=\frac{-(-1)(e^{-x})}{(1&plus;e^{-x})^2}=\frac{e^{-x}}{(1&plus;e^{-x})^2}&space;\newline&space;&=\left(\frac{1}{1&plus;e^{-x}}\right)\left(\frac{e^{-x}}{1&plus;e^{-x}}\right)=\sigma(x)\left(\frac{&plus;1-1&space;&plus;&space;e^{-x}}{1&plus;e^{-x}}\right)=\sigma(x)\left(\frac{1&space;&plus;&space;e^{-x}}{1&plus;e^{-x}}&space;-&space;\frac{1}{1&plus;e^{-x}}\right)=\sigma(x)(1&space;-&space;\sigma(x))\end{align*}" title="\LARGE \begin{align*}\sigma(x)'&=\left(\frac{1}{1+e^{-x}}\right)'=\frac{-(1+e^{-x})'}{(1+e^{-x})^2}=\frac{-1'-(e^{-x})'}{(1+e^{-x})^2}=\frac{0-(-x)'(e^{-x})}{(1+e^{-x})^2}=\frac{-(-1)(e^{-x})}{(1+e^{-x})^2}=\frac{e^{-x}}{(1+e^{-x})^2} \newline &=\left(\frac{1}{1+e^{-x}}\right)\left(\frac{e^{-x}}{1+e^{-x}}\right)=\sigma(x)\left(\frac{+1-1 + e^{-x}}{1+e^{-x}}\right)=\sigma(x)\left(\frac{1 + e^{-x}}{1+e^{-x}} - \frac{1}{1+e^{-x}}\right)=\sigma(x)(1 - \sigma(x))\end{align*}" /></a>
