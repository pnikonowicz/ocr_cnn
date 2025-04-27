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

func RandomUniformDistrbutionFunc(min, max float64) func(int) float64 {
	return func(_ int) float64 {
		return min + (max-min)*rand.Float64()
	}
}

func NormalDistributionHe() func(int) float64 {
	return func(n int) float64 {
		if n <= 0 {
			PrintAndTerminate(fmt.Sprintf("invalid n vaue for NormalHe: %d", n))
		}
		stdDev := math.Sqrt(2.0 / float64(n))
		return stdDev * rand.NormFloat64()
	}
}

func ReLU(x float64) float64 {
	return math.Max(float64(x), float64(0))
}

func SoftMax(logits []float64) []float64 { // uses stable implementation
	maxLogit := logits[0]
	for _, v := range logits {
		maxLogit = max(maxLogit, v)
	}

	result := make([]float64, len(logits))
	exponentiation := float64(0)
	for i, logit := range logits {
		reduced := logit - maxLogit
		result[i] = math.Exp(reduced)
		exponentiation += result[i]
	}

	for i := range len(logits) {
		inputScoreLogit := result[i]
		result[i] = inputScoreLogit / exponentiation
	}

	return result
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
