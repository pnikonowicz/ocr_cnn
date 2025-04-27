package neuron

import (
	"image"
	"image/color"
	"maps"
	"ocr_cnn/pkg/common"
	"testing"
)

func TestGraphIsConnectedWithRandomWeightsAndBias(t *testing.T) {
	const randomNumber = 1.2
	randomFunc := func(fanInSize int) float64 {
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
			if neuron.Bias != 0 {
				t.Errorf("bias should be zero %f", neuron.Bias)
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
	outputBiasA := float64(.1)
	outputBiasB := float64(.2)
	hiddenBiasA := float64(.3)
	hiddenBiasB := float64(.4)

	inputActivationA := float64(.5)
	inputActivationB := float64(.6)

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

	weightInputAtoHiddenA := float64(.7)
	weightInputAtoHiddenB := float64(.8)
	weightInputBtoHiddenA := float64(.9)
	weightInputBtoHiddenB := float64(.01)
	weightHiddenAtoOutputA := float64(.02)
	weightHiddenAtoOutputB := float64(.03)
	weightHiddenBtoOutputA := float64(.04)
	weightHiddenBtoOutputB := float64(.05)

	{
		// connect network
		inputNeuronA.Output = []*Edge{
			{
				Weight: &Weight{Value: float64(weightInputAtoHiddenA)},
				Neuron: &hiddenNeuronA,
			},
			{
				Weight: &Weight{Value: float64(weightInputAtoHiddenB)},
				Neuron: &hiddenNeuronB,
			},
		}
		inputNeuronB.Output = []*Edge{
			{
				Weight: &Weight{Value: float64(weightInputBtoHiddenA)},
				Neuron: &hiddenNeuronA,
			},
			{
				Weight: &Weight{Value: float64(weightInputBtoHiddenB)},
				Neuron: &hiddenNeuronB,
			},
		}
		hiddenNeuronA.Output = []*Edge{
			{
				Weight: &Weight{Value: float64(weightHiddenAtoOutputA)},
				Neuron: &outputNeuronA,
			},
			{
				Weight: &Weight{Value: float64(weightHiddenAtoOutputB)},
				Neuron: &outputNeuronB,
			},
		}
		hiddenNeuronB.Output = []*Edge{
			{
				Weight: &Weight{Value: float64(weightHiddenBtoOutputA)},
				Neuron: &outputNeuronA,
			},
			{
				Weight: &Weight{Value: float64(weightHiddenBtoOutputB)},
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

	logits := []float64{
		(expectedHiddenActivationA * weightHiddenAtoOutputA) + (expectedHiddenActivationB * weightHiddenBtoOutputA) + outputBiasA,
		(expectedHiddenActivationA * weightHiddenAtoOutputB) + (expectedHiddenActivationB * weightHiddenBtoOutputB) + outputBiasB,
	}

	expectedActivations := map[*Neuron]float64{
		&inputNeuronA:  inputActivationA,
		&inputNeuronB:  inputActivationB,
		&outputNeuronA: common.SoftMax(logits)[0],
		&outputNeuronB: common.SoftMax(logits)[1],
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
			t.Fatalf("output activation %d: expected %f but got %f", i, expectedActivation, outputNode.Activation)
		}
	}

	for i, inputNode := range ann.InputLayer {
		expectedActivation := expectedActivations[inputNode]
		if inputNode.Activation != expectedActivation {
			t.Fatalf("input activation %d: expected %f but got %f", i, expectedActivation, inputNode.Activation)
		}
	}
}

func TestInputEncoding(t *testing.T) {
	expectedColorBlack := color.RGBA{0, 0, 0, 255}
	expectedColorWhite := color.RGBA{255, 255, 255, 255}

	neuronA := &Neuron{}
	neuronB := &Neuron{}
	neuronC := &Neuron{}
	neuronD := &Neuron{}

	ann := &ANN{
		InputLayer: []*Neuron{
			neuronA,
			neuronB,
			neuronC,
			neuronD,
		},
	}

	img := image.NewRGBA(image.Rect(0, 0, 2, 2))

	img.Set(0, 0, expectedColorBlack)
	img.Set(0, 1, expectedColorWhite)
	img.Set(1, 0, expectedColorWhite)
	img.Set(1, 1, expectedColorBlack)

	ann.InputEncoding(img)

	expectedActivations := map[*Neuron]float64{
		neuronA: 0,
		neuronB: 1,
		neuronC: 1,
		neuronD: 0,
	}

	for i, neuron := range ann.InputLayer {
		expectedActivation := expectedActivations[neuron]
		if expectedActivation != neuron.Activation {
			t.Errorf("expected pixel %f but received %f at neuron %d", expectedActivation, neuron.Activation, i)
		}
	}
}
