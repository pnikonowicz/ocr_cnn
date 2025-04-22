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
	for _, v := range(actual) {
		sum += v
	}

	if math.Abs(sum-1.0) > 1e-9 { // compare threshold delta for floats
		t.Errorf("expected sum to equal 1 but was: %f", sum)
	}
}
