package common

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
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
