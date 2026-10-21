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
