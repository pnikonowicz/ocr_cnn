package common

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"math"
	"math/rand/v2"
	"os"
	"path"
)

func Log(message string) {
	fmt.Println(message)
}

func PrintAndTerminate(message string) {
	Log(message)
	os.Exit(1)
}

func RandomUniformDistrbutionFunc(min, max float32) func() float32 {
	return func() float32 {
		return min + (max-min)*rand.Float32()
	}
}

func ReLU(x float32) float32 {
	return float32(math.Max(float64(x), float64(0)))
}

func SoftMax(inputScoreLogit float32, logits []float32) float32 {
	exponentiation := float64(0)

	for _, logit := range logits {
		exponentiation += math.Exp(float64(logit))
	}

	normalization := math.Exp(float64(inputScoreLogit)) / exponentiation

	return float32(normalization)
}

func GetImage(number string, idx int) image.Image {
	wd, _ := os.Getwd()
	dataset_dir := path.Join(wd, "translated_dataset")

	dir := path.Join(dataset_dir, number)
	dir_iterator, err := os.ReadDir(dir)

	if err != nil {
		PrintAndTerminate("could not read dir")
	}

	file_name := path.Join(dir, dir_iterator[idx].Name())
	file_contents, err := os.ReadFile(file_name)

	if err != nil {
		PrintAndTerminate(fmt.Sprintf("could not read file: %s", file_name))
	}

	img, err := png.Decode(bytes.NewReader(file_contents))
	if err != nil {
		PrintAndTerminate(fmt.Sprintf("could not read png: %s", file_name))
	}

	return img
}
