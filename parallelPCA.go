package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// generateDataset creates a synthetic dataset
func generateDataset(nSamples, nFeatures int) *mat.Dense {
	data := make([]float64, nSamples*nFeatures)
	for i := range data {
		data[i] = rand.Float64()
	}
	return mat.NewDense(nSamples, nFeatures, data)
}

// computeCovariance calculates the covariance matrix
func computeCovariance(X *mat.Dense) *mat.SymDense {
	var centered mat.Dense
	centered.CloneFrom(X)
	nSamples, nFeatures := X.Dims()

	for j := 0; j < nFeatures; j++ {
		col := mat.Col(nil, j, X)
		mean := mat.Sum(mat.NewVecDense(len(col), col)) / float64(nSamples)
		for i := 0; i < nSamples; i++ {
			centered.Set(i, j, X.At(i, j)-mean)
		}
	}

	var cov mat.SymDense
	cov.SymOuterK(1/(float64(nSamples)-1), &centered)
	return &cov
}

// eigenDecomposition performs eigen decomposition on the covariance matrix
func eigenDecomposition(cov *mat.SymDense) ([]float64, *mat.Dense) {
	var eigen mat.Eigen
	ok := eigen.Factorize(cov, mat.EigenRight)
	if !ok {
		panic("Eigendecomposition failed")
	}

	values := eigen.Values(nil)
	vectors := eigen.Vectors()

	// Sort eigenvalues and eigenvectors
	n := len(values)
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			if math.Abs(values[indices[i]]) < math.Abs(values[indices[j]]) {
				indices[i], indices[j] = indices[j], indices[i]
			}
		}
	}

	sortedValues := make([]float64, n)
	sortedVectors := mat.NewDense(n, n, nil)
	for i, idx := range indices {
		sortedValues[i] = values[idx]
		col := mat.Col(nil, idx, vectors)
		sortedVectors.SetCol(i, col)
	}

	return sortedValues, sortedVectors
}

// timePCA measures the execution time of PCA
func timePCA(numThreads int, X *mat.Dense, numRuns int) float64 {
	runtime.GOMAXPROCS(numThreads)
	totalTime := 0.0

	for i := 0; i < numRuns; i++ {
		start := time.Now()

		cov := computeCovariance(X)

		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			eigenDecomposition(cov)
		}()
		wg.Wait()

		elapsed := time.Since(start).Seconds()
		totalTime += elapsed
	}

	return totalTime / float64(numRuns)
}

func main() {
	datasetConfigs := [][2]int{
		{1000, 50}, {1000, 500}, {1000, 5000},
		{5000, 50}, {5000, 500}, {5000, 5000},
		{10000, 50}, {10000, 500}, {10000, 5000},
	}

	threadCounts := []int{1, 2, 4, 8, 16, 32, 64}
	results := make(map[[2]int]map[string][]plotter.XYs)

	for _, config := range datasetConfigs {
		nSamples, nFeatures := config[0], config[1]
		fmt.Printf("Processing dataset with %d samples and %d features...\n", nSamples, nFeatures)

		X := generateDataset(nSamples, nFeatures)

		executionTimes := make(plotter.XYs, len(threadCounts))
		speedups := make(plotter.XYs, len(threadCounts))

		singleThreadTime := timePCA(1, X, 5)

		for i, numThreads := range threadCounts {
			execTime := timePCA(numThreads, X, 5)
			speedup := singleThreadTime / execTime

			executionTimes[i].X = float64(numThreads)
			executionTimes[i].Y = execTime
			speedups[i].X = float64(numThreads)
			speedups[i].Y = speedup
		}

		results[config] = map[string][]plotter.XYs{
			"execution_times": {executionTimes},
			"speedups":        {speedups},
		}
	}

	// Plotting results
	for _, config := range datasetConfigs {
		nSamples, nFeatures := config[0], config[1]

		p, err := plot.New()
		if err != nil {
			panic(err)
		}

		p.Title.Text = fmt.Sprintf("Execution Time (%d samples x %d features)", nSamples, nFeatures)
		p.X.Label.Text = "Number of Threads"
		p.Y.Label.Text = "Execution Time (seconds)"

		err = plotutil.AddLinePoints(p, "Execution Time", results[config]["execution_times"][0])
		if err != nil {
			panic(err)
		}

		if err := p.Save(4*vg.Inch, 4*vg.Inch, fmt.Sprintf("execution_time_%d_%d.png", nSamples, nFeatures)); err != nil {
			panic(err)
		}

		p, err = plot.New()
		if err != nil {
			panic(err)
		}

		p.Title.Text = fmt.Sprintf("Speedup (%d samples x %d features)", nSamples, nFeatures)
		p.X.Label.Text = "Number of Threads"
		p.Y.Label.Text = "Speedup"

		err = plotutil.AddLinePoints(p, "Speedup", results[config]["speedups"][0])
		if err != nil {
			panic(err)
		}

		if err := p.Save(4*vg.Inch, 4*vg.Inch, fmt.Sprintf("speedup_%d_%d.png", nSamples, nFeatures)); err != nil {
			panic(err)
		}
	}
}
