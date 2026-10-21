package main

import (
	"math"
	"math/rand"
)

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 - math.Exp(-x))
}

type Layer struct {
	Weights [][]float64
	Biases  []float64

	lastIn  []float64
	lastOut []float64
}

func NewLayer(nIn, nOut int) *Layer {
	w := make([][]float64, nOut)

	for i := range w {
		w[i] = make([]float64, nIn)
		for j := range w[i] {
			w[i][j] = rand.NormFloat64()
		}
	}

	return &Layer{
		Weights: w, Biases: make([]float64, nOut),
	}
}

func (l *Layer) Forward(in []float64) []float64 {
	l.lastIn = in
	out := make([]float64, len(l.Weights))

	for i := range l.Weights {
		sum := l.Biases[i]
		for j := range in {
			sum += l.Weights[i][j] * in[j]
		}
		out[i] = sigmoid(sum)
	}

	l.lastOut = out
	return out
}

func (l *Layer) Backward(gradOut []float64, lr float64) []float64 {
	gradIn := make([]float64, len(l.lastIn))

	for i := range l.Weights {
		delta := gradOut[i] * l.lastOut[i] * (1 - l.lastOut[i])
		for j := range l.lastIn {
			gradIn[j] += delta * l.Weights[i][j]
			l.Weights[i][j] -= lr * delta
		}
		l.Biases[i] -= lr * delta
	}
	return gradIn
}
