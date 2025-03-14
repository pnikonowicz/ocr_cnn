# Optical Character Recognition via a Convolution Neural Network

## Files

We are only trying to classify individual numbers.

### cmd/translate_dataset/main.go

Image Binarization: Takes the dataset and converts to only be black and white.

### cmd/verify_dataset/main.go

Asserts that the images are formatted properly.
All images must have the same resolution and only contain 2 colors.
