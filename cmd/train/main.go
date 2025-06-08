package main

import (
	"bytes"
	"fmt"
	"image"
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

	loss := float64(0)
	for i := 0; i <= 9; i++ {
		common.Debug(fmt.Sprintf("loading image %d", i))
		img := common.GetImage(fmt.Sprintf("%d", i), 0)
		loss += singlePassWithImage(&ann, img, i)
	}
	mean := loss / 10

	common.Log(fmt.Sprintf("mean: %f", mean))
	common.Log("done")
}

func singlePassWithImage(ann *neuron.ANN, img image.Image, imageType int) float64 {
	common.Debug("encode image")
	ann.InputEncoding(img)

	common.Debug("forward propagate")
	ann.ForwardPropagation()

	common.Debug("output layer")

	expectedOneHotEncoding := make([]float64, 10) // 10 possible images
	expectedOneHotEncoding[imageType] = 1         // onehot encoding value maps to imageType

	sum := float64(0)
	for i, node := range ann.OutputLayer {
		scalar := node.Activation
		sum += scalar
		common.Debug(fmt.Sprintf("   %d: %f", i, scalar))
	}
	common.Debug(fmt.Sprintf("sum (should equal 100): %f", sum))

	loss := common.CrossEntropyLoss(expectedOneHotEncoding, outputToVector(ann.OutputLayer))
	common.Debug(fmt.Sprintf("loss for image %d: %f", imageType, loss))

	learningRate := .01
	ann.BackwardPropagation(expectedOneHotEncoding, learningRate)

	return loss
}

func outputToVector(neuron []*neuron.Neuron) []float64 {
	vector := make([]float64, len(neuron))

	for i := range len(neuron) {
		vector[i] = neuron[i].Activation
	}

	return vector
}
