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
	previousLayerMap := map[*Neuron]bool{nil: true}

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

			for _, previousEdge := range neuron.NeighborsFrom {
				if previousEdge.Weight != randomNumber {
					t.Errorf("previous edge weight should be random %f", previousEdge.Weight)
				}
			}
			for _, nextEdge := range neuron.NeighborsTo {
				if nextEdge.Weight != randomNumber {
					t.Errorf("next edge weight should be random %f", nextEdge.Weight)
				}
			}

			nextPreviousLayerMap[neuron] = true
		}

		nextLayerMap := map[*Neuron]bool{}
		{ // setup next layer
			for _, nextEdge := range currentLayer[0].NeighborsTo {
				nextLayerMap[nextEdge.Neuron] = true
			}
		}
		{ // assert that every next edge points to the same set of next layer neurons
			for _, neuron := range currentLayer {
				for _, edge := range neuron.NeighborsTo {
					_, nextNeuronExists := nextLayerMap[edge.Neuron]
					if !nextNeuronExists {
						t.Errorf("a neuron for the next layer was not shared. all edges from one layer should point to the same set of neurons in the next layer")
					}
				}
			}
		}
		{ // assert that all previous edges points to the same set of previous layer neurons
			for _, neuron := range currentLayer {
				for _, previousEdge := range neuron.NeighborsFrom {
					_, previousNeuronExists := previousLayerMap[previousEdge.Neuron]
					if !previousNeuronExists {
						t.Errorf("could not find the neuron in previous layer. should be connected")
					}
				}
			}
		}

		previousLayerMap = nextPreviousLayerMap
		currentLayer = []*Neuron{}
		{ // setup the next layer
			for neuron := range maps.Keys(nextLayerMap) {
				currentLayer = append(currentLayer, neuron)
			}
		}
	}
}
