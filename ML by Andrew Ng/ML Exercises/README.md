### Assignments and Learning Points

## Week 1

Learning Points:
1. Cost function implementation
2. Feature normalization implementation
3. Gradient descent step implementation
4. Selecting learning rates

**Cost function, J**

`J = 1/2m * sum(sqrErrors), where sqrErrors = (X*theta - y).^2`

**Feature normalize X**

`X = (X .- mean(X)) ./ std(X)`

**Gradient descent single step**

`theta = theta - alpha * partial_derivation(J) = theta - alpha * 1/m * X' * (X*theta - y) (Vectorized)`

**Learning rate**

Gradient descent converges slowly if learning rate is small, and does not converge/diverge if learning rate is too big.
