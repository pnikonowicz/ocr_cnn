package neuron

import (
	"math"
)

type Weight struct {
	Value float32
}

type Edge struct {
	Weight *Weight
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
	OutputLayer  []*Neuron
}

func CreateANN(randomFunc func() float32, inputLayerSize, numberOfHiddenLayers int) ANN {
	layerSizes := []int{}
	{ // plot the size of each layer
		layerSizes = append(layerSizes, inputLayerSize)
		for i := 1; i <= numberOfHiddenLayers; i++ {
			reductionDivisior := int(math.Pow(2, float64(i)))
			layerSizes = append(layerSizes, inputLayerSize / reductionDivisior)
		}
		layerSizes = append(layerSizes, 10)
	}

	var firstLayer []*Neuron
	lastLayer := []*Neuron{}

	for _, layerSize := range(layerSizes) {
		var currentLayer []*Neuron
		for range layerSize {
			currentLayer = append(currentLayer, &Neuron{
				Activation: 0,
				Bias: randomFunc(),
				Input: nil, // we fill this in below
				Output: nil, // we fill this in below
			})
		}

		for _, lastNeuron := range(lastLayer) { // connect the graph bipartite
			for _, currentNeuron := range(currentLayer) {
				weight := Weight {Value: randomFunc() }
				lastNeuron.Output = append(lastNeuron.Output, &Edge {
					Neuron: currentNeuron,
					Weight: &weight,
				})
				currentNeuron.Input = append(currentNeuron.Input, &Edge {
					Neuron: lastNeuron,
					Weight: &weight,
				})
			}
		}
		lastLayer = currentLayer
		if firstLayer == nil {
			firstLayer = currentLayer
		}
	}

	ann := ANN{
		InputLayer:   firstLayer,
		OutputLayer:  lastLayer,
	}

	return ann
}
