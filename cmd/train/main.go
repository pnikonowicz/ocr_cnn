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

	ann := neuron.CreateANN(common.RandomUniformDistrbutionFunc(-1, 1), layerSize, numberOfHiddenLayers)

	fmt.Printf("created %d input layer neurons\n", len(ann.InputLayer))
	firstHiddenLayer := map[*neuron.Neuron]bool{}
	for _, edge := range ann.InputLayer[0].Output {
		firstHiddenLayer[edge.Neuron] = true
	}

	common.Log(fmt.Sprintf("created %d first hidden layer neurons", len(ann.InputLayer[0].Output)))
	common.Log(fmt.Sprintf("created %d second hidden layer neurons", len(ann.OutputLayer[0].Input)))
	common.Log(fmt.Sprintf("created %d output layer neurons", len(ann.OutputLayer)))

	common.Log("loading an image")
	img := common.GetImage("0", 0)

	common.Log("encode image")
	ann.InputEncoding(img)

	common.Log("done")
}
