package neuron

import "math"

type Edge struct {
	Weight     float32
	NeuronTo   *Neuron
	NeuronFrom *Neuron
}

type Neuron struct {
	Neighbors  []*Edge
	Bias       float32
	Activation float32
}

type ANN struct {
	InputLayer   []*Neuron
	HiddenLayers [][]*Neuron
	OutputLayer  []*Neuron
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
		outputLayer = append(outputLayer, &Neuron{})
	}

	ann := ANN{
		InputLayer:   inputLayer,
		HiddenLayers: hiddenLayers,
		OutputLayer:  outputLayer,
	}

	return ann
}
