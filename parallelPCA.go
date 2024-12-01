package main

import (
    "fmt"
    "gonum.org/v1/gonum/mat"
    "gonum.org/v1/gonum/stat"
    "math"
    "sync"
    "time"
)

type PCA struct {
    components  *mat.Dense
    explained   []float64
    nComponents int
}

func NewPCA(nComponents int) *PCA {
    return &PCA{
        nComponents: nComponents,
    }
}

func (p *PCA) fitTransform(X *mat.Dense, nWorkers int) (*mat.Dense, error) {
    rows, cols := X.Dims()
    
    // Center the data
    centered := mat.NewDense(rows, cols, nil)
    means := make([]float64, cols)
    
    var wg sync.WaitGroup
    chunks := rows / nWorkers
    
    for w := 0; w < nWorkers; w++ {
        wg.Add(1)
        go func(worker int) {
            defer wg.Done()
            start := worker * chunks
            end := start + chunks
            if worker == nWorkers-1 {
                end = rows
            }
            
            for j := 0; j < cols; j++ {
                col := mat.Col(nil, j, X)
                means[j] = stat.Mean(col, nil)
                for i := start; i < end; i++ {
                    centered.Set(i, j, X.At(i, j) - means[j])
                }
            }
        }(w)
    }
    wg.Wait()

    // Compute covariance matrix
    var cov mat.Dense
    cov.Mul(centered.T(), centered)
    cov.Scale(1/float64(rows-1), &cov)

    // Compute eigenvalues and eigenvectors
    var eigen mat.EigenSym
    ok := eigen.Factorize(&cov, true)
    if !ok {
        return nil, fmt.Errorf("eigendecomposition failed")
    }

    // Get principal components
    vectors := eigen.Vectors()
    values := eigen.Values(nil)
    
    // Sort eigenvalues and eigenvectors
    p.components = mat.NewDense(cols, p.nComponents, nil)
    p.explained = make([]float64, p.nComponents)
    
    totalVar := 0.0
    for _, val := range values {
        totalVar += val
    }
    
    for i := 0; i < p.nComponents; i++ {
        p.explained[i] = values[i] / totalVar
        for j := 0; j < cols; j++ {
            p.components.Set(j, i, vectors.At(j, i))
        }
    }

    // Transform data
    var transformed mat.Dense
    transformed.Mul(centered, p.components)
    
    return &transformed, nil
}

func main() {
    // Generate sample data
    rows, cols := 1000, 100
    data := mat.NewDense(rows, cols, nil)
    for i := 0; i < rows; i++ {
        for j := 0; j < cols; j++ {
            data.Set(i, j, rand.NormFloat64())
        }
    }

    // Benchmark different worker counts
    workerCounts := []int{1, 2, 3, 4, 5, 6, 7, 8}
    times := make([]float64, len(workerCounts))

    for i, workers := range workerCounts {
        pca := NewPCA(2)
        start := time.Now()
        _, err := pca.fitTransform(data, workers)
        if err != nil {
            fmt.Printf("Error with %d workers: %v\n", workers, err)
            continue
        }
        times[i] = time.Since(start).Seconds()
    }

    // Print results
    fmt.Println("Worker count | Time (seconds)")
    fmt.Println("------------------------")
    for i, workers := range workerCounts {
        fmt.Printf("%11d | %12.2f\n", workers, times[i])
    }
}
