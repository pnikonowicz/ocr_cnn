package neuron

import (
	"maps"
	"testing"
)

func TestGraphIsConnectedWithRandomWeightsAndBias(t *testing.T) {
	const randomNumber = 1.2
	randomFunc := func() float32 {
		return randomNumber
	}
	ann := CreateANN(randomFunc, 2, 1)
	expectedLayerSizes := []int{2, 1, 10}

	currentLayer := ann.InputLayer

	if len(currentLayer) == 0 {
		t.Errorf("first layer should have neurons")
		return
	}

	for len(currentLayer) > 0 {
		nextLayerMap := map[*Neuron]bool{}

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
				if previousEdge.Weight.Value != randomNumber {
					t.Errorf("previous edge weight should be random %f", previousEdge.Weight)
				}
			}
			for _, nextEdge := range neuron.Output {
				if nextEdge.Weight.Value != randomNumber {
					t.Errorf("next edge weight should be random %f", nextEdge.Weight)
				}
			}
			for _, nextNeuron := range neuron.Output {
				nextLayerMap[nextNeuron.Neuron] = true
			}
		}

		var expectedLayerSize int
		expectedLayerSize, expectedLayerSizes = expectedLayerSizes[0], expectedLayerSizes[1:]
		if len(currentLayer) != expectedLayerSize {
			t.Errorf("layer size %d does not match expected size %d", len(nextLayerMap), expectedLayerSize)
			return
		}

		currentLayer = []*Neuron{}
		for neuron := range maps.Keys(nextLayerMap) { // setup next layer
			currentLayer = append(currentLayer, neuron)
		}
	}
}
