package main

/*
 Simulate the accuracy of a user's query (e.g without requiring PIR or encryptions)
 1) Sample hash functions randomly from the distribution
 2) Hash all dataset points under each hash function and construct hash tables
 3) For each query in the testing set
	a) Compute the NumProbes closest lattice points
	b) Starting with the smallest radii and closest probe, find the first nonempty colliding bucket
	c) Randomly choose an element if there are multiple to simulate capped buckets of size 1.
	d) Compute the distance of the returned answer from the query
	e) Compute the ratio with the distance to the query's true nearest neighbor
	f) Return "Hit" if less than c=2
*/

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/alexflint/go-arg"
	"github.com/sachaservan/private-ann/ann"
	"github.com/sachaservan/private-ann/hash"
	"github.com/sachaservan/vec"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

type ThreadRes struct {
	hits              int       // count of within cR of best
	misses            int       // count of outside cR of best
	neither           int       // count where no result is returned
	approximateRatios []float64 // ratio of distance/best distance

	collisionId   []int
	tableId       []int
	rawCollisions [][]uint32
}

func main() {
	var args struct {
		Dataset             string  `default:"../../../datasets/mnist"`
		Samples             int     `default:"10000"`
		Tables              int     `default:"1"`
		Probes              int     `default:"5"`
		PartitionFactor     float64 `default:"1"`
		Lattice             int     `default:"2"`
		ApproximationFactor float64 `default:"2"`
		Mode                string  `default:"train"`
		OutputDim           uint64  `default:"100"`
		CoordBits           uint64  `default:"16"`

		// a value large enough such that any translation will be random
		MaxCoordinateValue int `default:"1000"`

		// for normal sequence
		ProjectionWidthMean   float64 `default:"887.7"` // for MNIST
		ProjectionWidthStddev float64 `default:"244.9"` // for MNIST

		MinDistance float64 `default:"350"`
		MaxDistance float64 `default:"1500"`
	}

	arg.MustParse(&args)
	fmt.Printf("%+v\n", args)
	dataset := args.Dataset
	numTables := args.Tables
	numProbes := args.Probes
	coordBits := args.CoordBits
	probeValues := make([]int, 1)

	if numProbes > 0 {
		probeValues[0] = numProbes
	} else {
		// probeValues = []int{4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1
		probeValues = []int{500, 100, 50, 10, 5, 1}
	}

	c := args.ApproximationFactor

	data, testData, n, err := ann.ReadDataset(dataset)
	if err != nil {
		panic(err)
	}

	path := strings.Split(args.Dataset, "/")
	datasetName := path[len(path)-1]

	fmt.Printf("Read dataset\n")

	var testIndexes, testAnswers []int
	switch args.Mode {
	case "train":
		// We compute brute force answers for a sample of the training data
		// This is to avoid using the testing data to find parameters and thus overfitting
		// When we query, we are careful not to return the point as its own neighbor
		testIndexes, testAnswers, err = ReadTrainingTestData(datasetName, args.Samples)
		if err != nil {
			panic(err)
		}
		testData = nil
	case "test":
		testAnswers = make([]int, len(n))
		for i := range testAnswers {
			testAnswers[i] = n[i][0]
		}
	default:
		panic("Unrecognized mode")
	}

	n = nil

	radius := args.ProjectionWidthMean + 2*args.ProjectionWidthStddev

	inputDim := data[0].Size()
	outputDim := args.OutputDim
	tables := make([]*ann.HashTable, numTables)
	hashes := make([]hash.Hash, numTables)
	for i := 0; i < len(tables); i++ {
		hashes[i] = hash.NewMultiLatticeHash(inputDim, 2, radius, float64(args.MaxCoordinateValue))
	}

	fmt.Printf("Constructed hash functions\n")
	hashSize := uint64(math.Ceil(math.Log2(float64(len(data)))))
	fmt.Printf("Universal hashing down to %v\n", hashSize)
	for i := 0; i < len(tables); i++ {
		tables[i] = ann.NewHashTable(i, hashSize)
		tables[i].AddAll(hashes[i], data)
		fmt.Printf("%v: %v\n", radius, tables[i].Len())
	}

	fmt.Println("Finding max of JL-transformed coordinates")
	jlTransform := hash.NewHashCommon(inputDim, int(outputDim), 0, true)
	maxValue := 0.0
	for i := 0; i < len(data); i++ {
		pCompressed := jlTransform.Project(data[i])
		for j := 0; j < len(pCompressed.Coords); j++ {
			if math.Abs(pCompressed.Coords[j]) > maxValue {
				maxValue = math.Abs(pCompressed.Coords[j])
			}
		}
	}

	fmt.Printf("Max value %v\n", maxValue)

	numThreads := runtime.NumCPU()

	done := make(chan bool)

	sections := hash.Spans(len(testAnswers), numThreads)
	cache := make([][][]uint64, len(testAnswers))

	for iter, numProbes := range probeValues {
		directoryName := fmt.Sprintf("%v%vd%vx%v", datasetName, args.Lattice, numTables, numProbes)
		// creates the directory if doesn't exist
		err = os.MkdirAll(directoryName, 0700)
		if err != nil {
			panic(err)
		}

		buckets := ann.NewPBRBuckets(hash.Prime, uint64(float64(numProbes)*args.PartitionFactor))
		fmt.Printf("probes: %v partitions: %v\n", numProbes, buckets.NumBuckets)

		results := make([]ThreadRes, numThreads)
		for i := 0; i < numThreads; i++ {
			go func(i int) {
				t := &results[i]
				for row := sections[i][0]; row < sections[i][1]; row++ {
					var query *vec.Vec
					var queryIndex int
					switch args.Mode {
					case "train":
						queryIndex = testIndexes[row]
						query = data[queryIndex]
					case "test":
						query = testData[row]
						queryIndex = -1 // never reject collisions
					}
					if iter == 0 {
						cache[row] = make([][]uint64, len(tables))
						// to give some sense of progress
						if (row & 1023) == 1000 {
							fmt.Printf("completed query %v of %v\n", row-sections[i][0], sections[i][1]-sections[i][0])
						}
					}
					collisions, radius := SimulateQuery(tables, hashes, cache[row], query, queryIndex, numProbes, buckets)
					t.tableId = append(t.tableId, radius)
					t.rawCollisions = append(t.rawCollisions, collisions)
					if len(collisions) == 0 {
						t.neither++
						// ensure results line up between trials
						t.approximateRatios = append(t.approximateRatios, math.NaN())
						t.collisionId = append(t.collisionId, -1)
						continue
					}

					// find the best candidate
					found := false
					foundID := uint32(0)
					bestCompressedDist := math.MaxFloat64
					for _, id := range collisions {
						resultDistCompressed := TruncatedAndCompressedEuclideanDistance(
							query, data[id], jlTransform, maxValue, int(coordBits))

						if resultDistCompressed < bestCompressedDist {
							found = true
							foundID = id
							bestCompressedDist = resultDistCompressed
						}
					}

					// actual best distance
					bestDist := vec.EuclideanDistance(query, data[testAnswers[row]])
					trueApproximationRatio := math.MaxFloat64
					if found {
						resultDist := vec.EuclideanDistance(query, data[foundID])
						trueApproximationRatio = resultDist / bestDist
					}

					if trueApproximationRatio <= c {
						t.hits++
					} else {
						t.misses++
					}
				}
				done <- true
			}(i)
		}

		for i := 0; i < numThreads; i++ {
			<-done
		}
		hits := 0
		misses := 0
		neither := 0
		ideal := 0
		approximationRatio := make([]float64, 0)
		bestApproximateRatios := make([]float64, 0)

		collisions := make([]int, 0)
		collisionTables := make([]int, 0)
		rawCollisions := make([][]uint32, 0)
		for i := 0; i < numThreads; i++ {
			r := &results[i]
			hits += r.hits
			misses += r.misses
			neither += r.neither
			approximationRatio = append(approximationRatio, r.approximateRatios...)
			collisions = append(collisions, r.collisionId...)
			collisionTables = append(collisionTables, r.tableId...)
			rawCollisions = append(rawCollisions, r.rawCollisions...)
		}
		f, err := os.OpenFile(directoryName+"/"+"results.txt",
			os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			panic(err)
		}
		t := time.Now()
		f.WriteString("\n" + t.String() + "\n")
		f.WriteString(fmt.Sprintf("%+v\n", args))
		res := fmt.Sprintf("Hits: %v\nMisses: %v\nNeither: %v\nIdeal %v", hits, misses, neither, ideal)
		f.WriteString(res)
		fmt.Print(res)
		f.WriteString(fmt.Sprintf("%v\n", approximationRatio))

		histPlot(approximationRatio, directoryName+"/ApproxRatios", 100)
		histPlot(bestApproximateRatios, directoryName+"/BestApproxRatios", 100)

		f, err = os.Create(fmt.Sprintf("%v/results%d.json", directoryName, t.Unix()))
		if err != nil {
			panic(err)
		}
		tablesSizes := make([]int, len(tables))
		for i := range tables {
			tablesSizes[i] = tables[i].Len()
		}
		b, err := json.Marshal(ResultsStruct{
			Dataset:               datasetName,
			Location:              args.Dataset,
			Samples:               len(testIndexes),
			Mode:                  args.Mode,
			Tables:                numTables,
			Probes:                numProbes,
			Lattice:               args.Lattice,
			ApproximationFactor:   args.ApproximationFactor,
			ProjectionWidthMean:   args.ProjectionWidthMean,
			ProjectionWidthStddev: args.ProjectionWidthStddev,
			MaxCoordinateValue:    args.MaxCoordinateValue,
			MinDistance:           args.MinDistance,
			MaxDistance:           args.MaxDistance,
			Time:                  t,

			Hits:                hits,
			Misses:              misses,
			Neither:             neither,
			TableSizes:          tablesSizes,
			ApproximationRatios: approximationRatio,

			CollisionsIds:     collisions,
			CollisionTableIds: collisionTables,
			RawCollisions:     rawCollisions,
		})
		if err != nil {
			panic(err)
		}
		_, err = f.WriteString(string(b))
		if err != nil {
			panic(err)
		}
	}
}

// Euclidean distance where the points are first compressed using JL-transform
// and then truncated to truncBits per coordinate
func TruncatedAndCompressedEuclideanDistance(
	p, q *vec.Vec,
	jlTransform *hash.HashCommon,
	maxValue float64,
	truncBits int) float64 {

	if p.Size() != q.Size() {
		panic("points must have the same dimentions")
	}

	pCompressed := jlTransform.Project(p)
	qCompressed := jlTransform.Project(q)

	distance := 0.0
	for i := 0; i < len(pCompressed.Coords); i++ {
		coordTruncP := pCompressed.Coords[i]
		coordTruncQ := qCompressed.Coords[i]
		coordTruncP = (coordTruncP / maxValue)
		coordTruncQ = (coordTruncQ / maxValue)
		negativeP := coordTruncP < 0
		negativeQ := coordTruncQ < 0

		coordTruncP = math.Abs(coordTruncP) // convert to fix point
		coordTruncQ = math.Abs(coordTruncQ) // convert to fix point
		coordTruncP *= float64(int(1<<truncBits) - 1)
		coordTruncQ *= float64(int(1<<truncBits) - 1)
		coordTruncIntP := int(coordTruncP)
		coordTruncIntQ := int(coordTruncQ)

		if negativeP {
			coordTruncIntP = -coordTruncIntP
		}

		if negativeQ {
			coordTruncIntQ = -coordTruncIntQ
		}

		// fmt.Printf("%v %v\n", coordTrunc, coordTruncInt)
		distance += math.Pow(float64(coordTruncIntP-coordTruncIntQ), 2)
	}

	return math.Sqrt(distance)
}

func SimulateQuery(tables []*ann.HashTable, hashes []hash.Hash, cache [][]uint64, query *vec.Vec, queryId int, probes int, buckets *ann.PBRBuckets) ([]uint32, int) {
	res := make([]uint32, 0)
	for i := range tables {
		bucketsUsed := make([]bool, buckets.NumBuckets)
		if cache[i] == nil {
			cache[i] = hashes[i].MultiHash(query, probes)
		}
		hashes := cache[i][:probes]
		for _, h := range hashes {
			bucket := buckets.FindBucket(h)
			if bucketsUsed[bucket] {
				continue
			}
			bucketsUsed[bucket] = true
			collisions := tables[i].Get(h)
			if len(collisions) > 1 {
				// choose the random member kept from the capped bucket
				r := rand.Intn(len(collisions))
				if collisions[r] == uint32(queryId) {
					// len(collisions) is at least two in this branch so there is always another choice
					r = (r + 1) % len(collisions)
				}
				res = append(res, collisions[r])
			} else if len(collisions) == 1 {
				if collisions[0] != uint32(queryId) {
					res = append(res, collisions[0])
				}
			}
		}
	}
	return res, len(tables)
}

func ReadTrainingTestData(datasetName string, numTests int) ([]int, []int, error) {
	f, err := os.Open("../meanAndStd/" + datasetName + ".txt")
	if err != nil {
		return nil, nil, err
	}
	testIndexes := make([]int, numTests)
	testAnswers := make([]int, numTests)
	r := bufio.NewReader(f)
	for i := 0; i < numTests; i++ {
		s, err := r.ReadString('\n')
		if err != io.EOF && err != nil {
			panic(err)
		}
		parts := strings.Split(s, ":")
		if len(parts) != 2 {
			i--
			continue
		}
		var e error
		testIndexes[i], e = strconv.Atoi(strings.TrimSpace(parts[0]))
		if e != nil {
			panic(e)
		}
		testAnswers[i], e = strconv.Atoi(strings.TrimSpace(parts[1]))
		if e != nil {
			panic(e)
		}

		if err == io.EOF && i != numTests-1 {
			return testIndexes, testAnswers, errors.New("Data ended early!")
		}
	}
	return testIndexes, testAnswers, nil
}

func histPlot(values []float64, title string, numBins int) {
	for i, v := range values {
		if v < 0 {
			v = 0
		}
		if math.IsNaN(v) || math.IsInf(v, 0) {
			v = 0
		}
		values[i] = v
	}
	p := plot.New()
	p.Title.Text = title
	hist, err := plotter.NewHist(plotter.Values(values), numBins)
	if err != nil {
		panic(err)
	}
	p.Add(hist)

	if err := p.Save(3*vg.Inch, 3*vg.Inch, title+".png"); err != nil {
		panic(err)
	}
}

type ResultsStruct struct {
	Dataset  string
	Location string
	Samples  int
	Mode     string

	Tables              int
	Probes              int
	Lattice             int
	ApproximationFactor float64
	SequenceType        string
	Time                time.Time

	// a value large enough such that any translation will be random
	MaxCoordinateValue int

	// for normal sequence
	ProjectionWidthMean   float64
	ProjectionWidthStddev float64

	MinDistance float64
	MaxDistance float64

	Hits                int
	Misses              int
	Neither             int
	Radii               []float64
	TableSizes          []int
	CollisionsIds       []int
	CollisionTableIds   []int
	ApproximationRatios []float64
	RawCollisions       [][]uint32
}
