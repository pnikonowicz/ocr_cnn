package common

import (
	"fmt"
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

func RandomUniformDistrbution(min, max float32) func() float32 {
	return func() float32 {
		return min + (max-min)*rand.Float32()
	}
}
