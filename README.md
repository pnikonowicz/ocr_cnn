# Optical Character Recognition via an Artificial Neural Network

Uses one hot encoding to translate a list of 10 arabic numeral images (0-9) and classify them into the cooresponding digit.

Forward propagation uses ReLU for each hidden layer activation function and Softmax for the output layer.

All layers are bipartite.

## Files

We are only trying to classify individual numbers.

### cmd/translate_dataset/main.go

Image Binarization: Takes the dataset and converts to only be black and white.

### cmd/verify_dataset/main.go

Asserts that the images are formatted properly.
All images must have the same resolution and only contain 2 colors.

## Development

run all tests with:

```bash
go test ./...
```
