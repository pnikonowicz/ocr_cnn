# TODO

## forward propagation bug

1. it appears as if we are overflowing on forward propagation
2. output layer probabilities should sum to 100% but do not

## back propagation

### loss function

* use categorical cross-entropy on the output layer

### loss function mean

* apply the loss function for X amount of test runs and find the mean value

### update weights

* compute the gradient of this mean loss with respect to each weight in the output layer
* continue the process (chain rule) going backward
