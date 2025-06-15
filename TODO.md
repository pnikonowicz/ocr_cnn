# TODO

## back propagation

* use logits, softmax to update weights and biases

### update weights

* compute the gradient of this mean loss with respect to each weight in the output layer
* continue the process (chain rule) going backward

## refactor neuron_test setup

* create BDD style to reuse setup for creating the ANN

## refactor test weight value pattern

utilize the pattern `weightInputAtoOutputA := Weight{Value: float64(.7)}`
for reusing weights

## refactor test values

currently test values are in ascending order (eg: .1, .2, .3). but when we move things around,
the order is tedious to maintain. 

devise a way to allow the code lines to be able to move around while maintaining the relative 
increasing values (maybe a function? maybe a pattern? something else?)