package neuron

import (
	"math"
	"math/rand"
)

type Edge struct {
	Weight float32
	Neuron *Neuron
}

type Neuron struct {
	Input      []*Edge
	Output     []*Edge
	Bias       float32
	Activation float32
}

type ANN struct {
	InputLayer   []*Neuron
	HiddenLayers [][]*Neuron
	OutputLayer  []*Neuron
	RandomFunc   func() float32
}

func CreateANN(layerSize, numberOfHiddenLayers int) ANN {
	var inputLayer []*Neuron
	var hiddenLayers [][]*Neuron
	var outputLayer []*Neuron

	for range layerSize {
		inputLayer = append(inputLayer, &Neuron{})
	}

	for i := 1; i <= numberOfHiddenLayers; i++ {
		reductionDivisior := int(math.Pow(2, float64(i)))
		var hiddenLayer []*Neuron
		for range len(inputLayer) / reductionDivisior {
			hiddenLayer = append(hiddenLayer, &Neuron{})
		}
		hiddenLayers = append(hiddenLayers, hiddenLayer)
	}

	for i := 0; i < 10; i++ {
		var neighborEdges []*Edge

		outputLayer = append(outputLayer, &Neuron{
			Activation: 0.0,
			Output:     nil,
			Input:      neighborEdges,
		})
	}

	ann := ANN{
		InputLayer:   inputLayer,
		HiddenLayers: hiddenLayers,
		OutputLayer:  outputLayer,
		RandomFunc:   rand.Float32,
	}

	return ann
}
