package common

import (
	"math"
	"testing"
)

func TestSoftMax(t *testing.T) {
	expected := math.Exp(2) / (math.Exp(2) + math.Exp(3) + math.Exp(4))
	actual := SoftMax([]float64{2, 3, 4})

	if actual[0] != float64(expected) {
		t.Errorf("expected %f but was %f", expected, actual)
	}

	// should add up to near 1
	sum := float64(0)
	for _, v := range actual {
		sum += v
	}

	if math.Abs(sum-1.0) > 1e-9 { // compare threshold delta for floats
		t.Errorf("expected sum to equal 1 but was: %f", sum)
	}
}

func TestCrossEntropyLoss(t *testing.T) {
	trueProability := []float64{0, 1, 0}
	predictedProbility := []float64{0.1, 0.8, 0.1}
	expectedLoss := -math.Log(0.8)

	actualLoss := CrossEntropyLoss(trueProability, predictedProbility)

	if actualLoss != expectedLoss {
		t.Errorf("expected loss to be %f but was: %f", expectedLoss, actualLoss)
	}
}
