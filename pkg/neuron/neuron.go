package neuron

import (
	"fmt"
	"image"
	"image/color"
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
	calculateHiddenLayerActivations := func(inputLayer []*Neuron) {
		currentLayer := inputLayer

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

	calculateOutputLayerActivations := func(outputLayer []*Neuron) {
		logits := make([]float32, len(outputLayer))

		for i, neuron := range outputLayer {
			logit := float32(0)
			for _, inputNeuron := range neuron.Input {
				activation := inputNeuron.Neuron.Activation
				weight := inputNeuron.Weight.Value

				logit += (activation * weight)

			}
			bias := neuron.Bias

			logits[i] = logit + bias
		}

		for i, neuron := range outputLayer {
			neuron.Activation = common.SoftMax(logits[i], logits)
		}
	}

	calculateHiddenLayerActivations(ann.InputLayer)
	calculateOutputLayerActivations(ann.OutputLayer)
}

func colorsEqual(c1, c2 color.Color) bool {
	r1, g1, b1, a1 := c1.RGBA()
	r2, g2, b2, a2 := c2.RGBA()
	return r1 == r2 && g1 == g2 && b1 == b2 && a1 == a2
}

func (ann *ANN) InputEncoding(img image.Image) {
	bounds := img.Bounds()

	for x := bounds.Min.X; x < bounds.Max.X; x++ {
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			neuronIdx := (x * bounds.Max.X) + y
			neuron := ann.InputLayer[neuronIdx]

			if colorsEqual(color.Black, img.At(x, y)) {
				neuron.Activation = 0
			} else {
				neuron.Activation = 1
			}
		}
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
