package neuron

import (
	"fmt"
	"math"
	"ocr_cnn/pkg/common"
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
	InputLayer  []*Neuron
	OutputLayer []*Neuron
}

func CreateANN(randomFunc func() float32, inputLayerSize, numberOfHiddenLayers int) ANN {
	layerSizes := []int{}
	{ // plot the size of each layer
		layerSizes = append(layerSizes, inputLayerSize)
		for i := 1; i <= numberOfHiddenLayers; i++ {
			reductionDivisior := int(math.Pow(2, float64(i)))
			layerSizes = append(layerSizes, inputLayerSize/reductionDivisior)
		}
		layerSizes = append(layerSizes, 10)
	}

	var firstLayer []*Neuron
	lastLayer := []*Neuron{}

	for _, layerSize := range layerSizes {
		var currentLayer []*Neuron
		for range layerSize {
			currentLayer = append(currentLayer, &Neuron{
				Activation: 0,
				Bias:       randomFunc(),
				Input:      nil, // we fill this in below
				Output:     nil, // we fill this in below
			})
		}

		for _, lastNeuron := range lastLayer { // connect the graph bipartite
			for _, currentNeuron := range currentLayer {
				weight := Weight{Value: randomFunc()}
				lastNeuron.Output = append(lastNeuron.Output, &Edge{
					Neuron: currentNeuron,
					Weight: &weight,
				})
				currentNeuron.Input = append(currentNeuron.Input, &Edge{
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
		InputLayer:  firstLayer,
		OutputLayer: lastLayer,
	}

	return ann
}

func (ann *ANN) ForwardPropagation() {
	currentLayer := ann.InputLayer

	for len(currentLayer) > 0 {
		activations := map[*Neuron]float32{}

		for _, inputNeuron := range currentLayer {
			for _, inputNeuronEdge := range inputNeuron.Output {
				outputNeuron := inputNeuronEdge.Neuron
				activations[outputNeuron] += (inputNeuronEdge.Weight.Value * inputNeuron.Activation)
			}
		}

		nextLayer := []*Neuron{}
		for outputNeuron, activation := range activations {
			outputNeuron.Activation = common.ReLU(activation + outputNeuron.Bias)
			nextLayer = append(nextLayer, outputNeuron)
		}

		currentLayer = nextLayer
	}
}

func (ann *ANN) Print(bindFunc func(string)) {
	currentLayer := ann.InputLayer

	for len(currentLayer) > 0 {
		nextLayer := map[*Neuron]bool{}
		currentLayerString := ""
		for _, node := range currentLayer {
			currentLayerString += fmt.Sprintf("Neuron(%f) | ", node.Activation)
			for _, nextNode := range node.Output {
				nextLayer[nextNode.Neuron] = true
			}
		}

		bindFunc(currentLayerString)

		currentLayer = []*Neuron{}
		for node := range nextLayer {
			currentLayer = append(currentLayer, node)
		}
	}
}
