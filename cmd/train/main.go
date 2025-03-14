package main

import (
	"fmt"
	"math"
	"ocr_cnn/pkg/neuron"
)

func findResolution() (int,int) {
	return 32,32
}

func main() {
	resolutionX, resolutionY := findResolution()
	layerSize := resolutionX * resolutionY
	const numberOfHiddenLayers = 2

	var inputLayer []neuron.Neuron
	var hiddenLayers [][]neuron.Neuron
	var outputLayer []neuron.Neuron

	for range(layerSize) {
		inputLayer = append(inputLayer, neuron.Neuron{})
	}

	for i := 1; i <= numberOfHiddenLayers; i++ {
		reductionDivisior := int(math.Pow(2, float64(i)))
		var hiddenLayer []neuron.Neuron
		for range(len(inputLayer) / reductionDivisior) {
			hiddenLayer = append(hiddenLayer, neuron.Neuron{})
		}
		hiddenLayers = append(hiddenLayers, hiddenLayer)
	}

	for i := 0; i<10; i++ {
		outputLayer = append(outputLayer, neuron.Neuron{})
	}

	fmt.Printf("created %d input layer neurons\n", len(inputLayer))
	fmt.Printf("created %d first hidden layer neurons\n", len(hiddenLayers[0]))
	fmt.Printf("created %d second hidden layer neurons\n", len(hiddenLayers[1]))
	fmt.Printf("created %d output layer neurons\n", len(outputLayer))
}