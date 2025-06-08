package neuron

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"ocr_cnn/pkg/common"
)

type Weight struct {
	Value float64
}

type Edge struct {
	Weight *Weight
	Neuron *Neuron
}

type Neuron struct {
	Input      []*Edge
	Output     []*Edge
	Bias       float64
	Activation float64
}

type ANN struct {
	InputLayer  []*Neuron
	OutputLayer []*Neuron
}

func CreateANN(randomFunc func(int) float64, inputLayerSize, numberOfHiddenLayers int) ANN {
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
				Bias:       0,
				Input:      nil, // we fill this in below
				Output:     nil, // we fill this in below
			})
		}

		for _, lastNeuron := range lastLayer { // connect the graph bipartite
			for _, currentNeuron := range currentLayer {
				weight := Weight{Value: randomFunc(len(lastLayer))}
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

func (ann *ANN) ForwardPropagation() []float64 {
	calculateHiddenLayerActivations := func(inputLayer []*Neuron) {
		currentLayer := inputLayer

		for len(currentLayer) > 0 {
			activations := map[*Neuron]float64{}

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

	calculateOutputLayerActivations := func(outputLayer []*Neuron) []float64 {
		logits := make([]float64, len(outputLayer))

		for i, neuron := range outputLayer {
			logit := float64(0)
			for _, inputNeuron := range neuron.Input {
				activation := inputNeuron.Neuron.Activation
				weight := inputNeuron.Weight.Value

				logit += float64(activation * weight)
			}

			bias := neuron.Bias
			logits[i] = logit + float64(bias)
		}

		for i, value := range common.SoftMax(logits) {
			outputLayer[i].Activation = value
		}

		return logits
	}

	calculateHiddenLayerActivations(ann.InputLayer)
	logits := calculateOutputLayerActivations(ann.OutputLayer)

	return logits
}

func (ann *ANN) BackwardPropagation(expectedOneHotEncoding []float64) {
	softmaxVector := outputToVector(ann.OutputLayer)

	// find softmax cross entropy gradient of loss w.r.t softmax
	gradientVector := common.SoftmaxCrossEntropyGradient(softmaxVector, expectedOneHotEncoding)
	fmt.Printf("gradient vector: %v\n", gradientVector)

	// find gradients via chain rule
	common.CrossEntropyPartialDerivative(softmaxVector, expectedOneHotEncoding)
	// Softmax partial derivitive
	// RELU gradiant for every hidden layer
}

func outputToVector(neuron []*Neuron) []float64 {
	vector := make([]float64, len(neuron))

	for i := range len(neuron) {
		vector[i] = neuron[i].Activation
	}

	return vector
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
