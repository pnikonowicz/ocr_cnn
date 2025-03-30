package common

import (
	"math"
	"testing"
)

func TestSoftMax(t *testing.T) {
	expected := math.Exp(2) / (math.Exp(2) + math.Exp(3) + math.Exp(4))
	actual := SoftMax(float32(2), []float32{2, 3, 4})

	if actual != float32(expected) {
		t.Errorf("expected %f but was %f", expected, actual)
	}
}
