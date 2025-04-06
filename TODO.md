# TODO

## read image and translate to input activation vector

given an image, convert values to the input vector

## back propagation

### loss function

* use categorical cross-entropy on the output layer

### loss function mean

* apply the loss function for X amount of test runs and find the mean value

### update weights

* compute the gradient of this mean loss with respect to each weight in the output layer
* continue the process (chain rule) going backward
