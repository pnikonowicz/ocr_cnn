package neuron

import (
	"maps"
	"ocr_cnn/pkg/common"
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

func TestForwardPropagationPeformsCorrectCalculations(t *testing.T) {
	outputBiasA := float32(.1)
	outputBiasB := float32(.2)
	hiddenBiasA := float32(.3)
	hiddenBiasB := float32(.4)

	inputActivationA := float32(.5)
	inputActivationB := float32(.6)

	inputNeuronA := Neuron{
		Activation: inputActivationA,
	}
	inputNeuronB := Neuron{
		Activation: inputActivationB,
	}
	hiddenNeuronA := Neuron{
		Bias: hiddenBiasA,
	}
	hiddenNeuronB := Neuron{
		Bias: hiddenBiasB,
	}
	outputNeuronA := Neuron{
		Bias: outputBiasA,
	}
	outputNeuronB := Neuron{
		Bias: outputBiasB,
	}

	weightInputAtoHiddenA := float32(.7)
	weightInputAtoHiddenB := float32(.8)
	weightInputBtoHiddenA := float32(.9)
	weightInputBtoHiddenB := float32(.01)
	weightHiddenAtoOutputA := float32(.02)
	weightHiddenAtoOutputB := float32(.03)
	weightHiddenBtoOutputA := float32(.04)
	weightHiddenBtoOutputB := float32(.05)

	{
		// connect network
		inputNeuronA.Output = []*Edge{
			{
				Weight: &Weight{Value: float32(weightInputAtoHiddenA)},
				Neuron: &hiddenNeuronA,
			},
			{
				Weight: &Weight{Value: float32(weightInputAtoHiddenB)},
				Neuron: &hiddenNeuronB,
			},
		}
		inputNeuronB.Output = []*Edge{
			{
				Weight: &Weight{Value: float32(weightInputBtoHiddenA)},
				Neuron: &hiddenNeuronA,
			},
			{
				Weight: &Weight{Value: float32(weightInputBtoHiddenB)},
				Neuron: &hiddenNeuronB,
			},
		}
		hiddenNeuronA.Output = []*Edge{
			{
				Weight: &Weight{Value: float32(weightHiddenAtoOutputA)},
				Neuron: &outputNeuronA,
			},
			{
				Weight: &Weight{Value: float32(weightHiddenAtoOutputB)},
				Neuron: &outputNeuronB,
			},
		}
		hiddenNeuronB.Output = []*Edge{
			{
				Weight: &Weight{Value: float32(weightHiddenBtoOutputA)},
				Neuron: &outputNeuronA,
			},
			{
				Weight: &Weight{Value: float32(weightHiddenBtoOutputB)},
				Neuron: &outputNeuronB,
			},
		}
		hiddenNeuronA.Input = []*Edge{
			{
				Weight: inputNeuronA.Output[0].Weight,
				Neuron: &inputNeuronA,
			},
			{
				Weight: inputNeuronB.Output[0].Weight,
				Neuron: &inputNeuronB,
			},
		}
		hiddenNeuronB.Input = []*Edge{
			{
				Weight: inputNeuronA.Output[1].Weight,
				Neuron: &inputNeuronA,
			},
			{
				Weight: inputNeuronB.Output[1].Weight,
				Neuron: &inputNeuronB,
			},
		}
		outputNeuronA.Input = []*Edge{
			{
				Weight: hiddenNeuronA.Output[0].Weight,
				Neuron: &hiddenNeuronA,
			},
			{
				Weight: hiddenNeuronB.Output[0].Weight,
				Neuron: &hiddenNeuronB,
			},
		}
		outputNeuronB.Input = []*Edge{
			{
				Weight: hiddenNeuronA.Output[1].Weight,
				Neuron: &hiddenNeuronA,
			},
			{
				Weight: hiddenNeuronB.Output[1].Weight,
				Neuron: &hiddenNeuronB,
			},
		}
	}

	expectedHiddenActivationA := common.ReLU((weightInputAtoHiddenA * inputActivationA) + (weightInputBtoHiddenA * inputActivationB) + hiddenBiasA)
	expectedHiddenActivationB := common.ReLU((weightInputBtoHiddenB * inputActivationB) + (weightInputAtoHiddenB * inputActivationA) + hiddenBiasB)

	logits := []float32{
		(expectedHiddenActivationA * weightHiddenAtoOutputA) + (expectedHiddenActivationB * weightHiddenBtoOutputA) + outputBiasA,
		(expectedHiddenActivationA * weightHiddenAtoOutputB) + (expectedHiddenActivationB * weightHiddenBtoOutputB) + outputBiasB,
	}

	expectedActivations := map[*Neuron]float32{
		&hiddenNeuronA: expectedHiddenActivationA,
		&hiddenNeuronB: expectedHiddenActivationB,
		&outputNeuronA: common.SoftMax(logits[0], logits),
		&outputNeuronB: common.SoftMax(logits[1], logits),
	}

	ann := &ANN{
		InputLayer: []*Neuron{
			&inputNeuronA, &inputNeuronB,
		},
		OutputLayer: []*Neuron{
			&outputNeuronA, &outputNeuronB,
		},
	}

	ann.ForwardPropagation()

	for i, outputNode := range ann.OutputLayer {
		expectedActivation := expectedActivations[outputNode]
		if outputNode.Activation != expectedActivation {
			t.Fatalf("activation %d: expected %f but got %f", i, expectedActivation, outputNode.Activation)
		}
	}
}
