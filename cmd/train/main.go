package main

import (
	"bytes"
	"fmt"
	"image/png"
	"ocr_cnn/pkg/common"
	"ocr_cnn/pkg/neuron"
	"os"
	"path"
)

func findResolution(dataset_dir string) (int, int) {
	file_path := path.Join(dataset_dir, "0", "Abadi_0.png")
	file_contents, err := os.ReadFile(file_path)
	if err != nil {
		common.PrintAndTerminate(fmt.Sprintf("could not read file: %s", file_path))
	}
	img, err := png.Decode(bytes.NewReader(file_contents))
	if err != nil {
		common.PrintAndTerminate(fmt.Sprintf("could not read png: %s", file_path))
	}

	bounds := img.Bounds()

	x := bounds.Max.X - bounds.Min.X
	y := bounds.Max.Y - bounds.Min.Y

	return x, y
}

func main() {
	wd, _ := os.Getwd()
	dataset_dir := path.Join(wd, "translated_dataset")

	resolutionX, resolutionY := findResolution(dataset_dir)
	layerSize := resolutionX * resolutionY
	const numberOfHiddenLayers = 2

	ann := neuron.CreateANN(common.NormalDistributionHe(), layerSize, numberOfHiddenLayers)

	common.Log(fmt.Sprintf("created %d input layer neurons", len(ann.InputLayer)))
	common.Log(fmt.Sprintf("created %d first hidden layer neurons", len(ann.InputLayer[0].Output)))
	common.Log(fmt.Sprintf("created %d second hidden layer neurons", len(ann.OutputLayer[0].Input)))
	common.Log(fmt.Sprintf("created %d output layer neurons", len(ann.OutputLayer)))

	imageNumber := "0"

	common.Log(fmt.Sprintf("loading image %s", imageNumber))
	img := common.GetImage(imageNumber, 0)

	common.Log("encode image")
	ann.InputEncoding(img)

	common.Log("forward propagate")
	ann.ForwardPropagation()

	common.Log("output layer")
	sum := float64(0)
	for i, neuron := range ann.OutputLayer {
		sum += neuron.Activation
		common.Log(fmt.Sprintf("   %d: %f", i, neuron.Activation))
	}
	common.Log(fmt.Sprintf("sum: %f", sum))

	trueProbabilities := []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	loss := common.CrossEntropyLoss(trueProbabilities, outputToVector(ann.OutputLayer))
	common.Log(fmt.Sprintf("loss for image %s: %f", imageNumber, loss))

	common.Log("done")
}

func outputToVector(neuron []*neuron.Neuron) []float64 {
	vector := make([]float64, len(neuron))

	for i := range len(neuron) {
		vector[i] = neuron[i].Activation
	}

	return vector
}
