package main

import (
	"bytes"
	"fmt"
	"image/png"
	"math"
	"ocr_cnn/pkg/neuron"
	"os"
	"path"
)

func log(message string) {
	fmt.Println(message)
}

func printAndTerminate(message string) {
	log(message)
	os.Exit(1)
}

func findResolution(dataset_dir string) (int, int) {
	file_path := path.Join(dataset_dir, "0", "Abadi_0.png")
	file_contents, err := os.ReadFile(file_path)
	if err != nil {
		printAndTerminate(fmt.Sprintf("could not read file: %s", file_path))
	}
	img, err := png.Decode(bytes.NewReader(file_contents))
	if err != nil {
		printAndTerminate(fmt.Sprintf("could not read png: %s", file_path))
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

	var inputLayer []neuron.Neuron
	var hiddenLayers [][]neuron.Neuron
	var outputLayer []neuron.Neuron

	for range layerSize {
		inputLayer = append(inputLayer, neuron.Neuron{})
	}

	for i := 1; i <= numberOfHiddenLayers; i++ {
		reductionDivisior := int(math.Pow(2, float64(i)))
		var hiddenLayer []neuron.Neuron
		for range len(inputLayer) / reductionDivisior {
			hiddenLayer = append(hiddenLayer, neuron.Neuron{})
		}
		hiddenLayers = append(hiddenLayers, hiddenLayer)
	}

	for i := 0; i < 10; i++ {
		outputLayer = append(outputLayer, neuron.Neuron{})
	}

	fmt.Printf("created %d input layer neurons\n", len(inputLayer))
	fmt.Printf("created %d first hidden layer neurons\n", len(hiddenLayers[0]))
	fmt.Printf("created %d second hidden layer neurons\n", len(hiddenLayers[1]))
	fmt.Printf("created %d output layer neurons\n", len(outputLayer))
}
