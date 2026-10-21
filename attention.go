package main

import "math"

func dotProduct(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func softmax(xs []float64) []float64 {
	res := make([]float64, len(xs))
	maxval := xs[0]
	for _, v := range xs {
		if v > maxval {
			maxval = v
		}
	}
	var sum float64
	for i, v := range xs {
		res[i] = math.Exp(v - maxval)
		sum += res[i]
	}

	for i := range res {
		res[i] /= sum
	}

	return res
}

func selfAttention(q []float64, keys, values [][]float64) []float64 {
	dK := float64(len(q))
	scores := make([]float64, len(keys))

	for i, k := range keys {
		scores[i] = dotProduct(q, k) / math.Sqrt(dK)
	}

	weights := softmax(scores)
	valDim := len(values[0])
	out := make([]float64, valDim)
	for i, w := range weights {
		for j := range valDim {
			out[j] += w * values[i][j]
		}
	}
	return out
}
