package neuron

import (
	"maps"
	"testing"
)

func TestGraphIsConnectedWithRandomWeightsAndBias(t *testing.T) {
	const randomNumber = 1.2
	ann := CreateANN(2, 1)

	ann.RandomFunc = func() float32 {
		return randomNumber
	}

	currentLayer := ann.InputLayer

	if len(currentLayer) == 0 {
		t.Errorf("first layer should have neurons")
		return
	}

	for len(currentLayer) > 0 {
		nextPreviousLayerMap := map[*Neuron]bool{}

		for _, neuron := range currentLayer { // assert that all values are random
			if neuron == nil {
				t.Errorf("current neuron should not be nil")
				return
			}
			if neuron.Activation != 0 {
				t.Errorf("activation value should start at zero: %f", neuron.Activation)
			}
			if neuron.Bias != randomNumber {
				t.Errorf("bias should be random %f", neuron.Bias)
			}

			for _, previousEdge := range neuron.Input {
				if previousEdge.Weight != randomNumber {
					t.Errorf("previous edge weight should be random %f", previousEdge.Weight)
				}
			}
			for _, nextEdge := range neuron.Output {
				if nextEdge.Weight != randomNumber {
					t.Errorf("next edge weight should be random %f", nextEdge.Weight)
				}
			}

			nextPreviousLayerMap[neuron] = true
		}

		nextLayerMap := map[*Neuron]bool{}
		{ // setup next layer
			for _, nextEdge := range currentLayer[0].Output {
				nextLayerMap[nextEdge.Neuron] = true
			}
		}

		currentLayer = []*Neuron{}
		{ // setup the next layer
			for neuron := range maps.Keys(nextLayerMap) {
				currentLayer = append(currentLayer, neuron)
			}
		}
	}
}
