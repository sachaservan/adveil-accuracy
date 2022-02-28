package ann

import (
	"log"
	"math/rand"
	"runtime"
	"sync"

	"github.com/sachaservan/private-ann/hash"
	"github.com/sachaservan/vec"
)

type HashTable struct {
	table  int
	mask   uint64
	hashes map[uint64][]uint32
	mu     sync.Mutex
}

func NewHashTable(table int, numBits uint64) *HashTable {
	mask := (uint64(1) << numBits) - uint64(1)
	return &HashTable{table: table, hashes: make(map[uint64][]uint32), mask: mask}
}

func ComputeHashes(n int, h hash.Hash, data []*vec.Vec, numBits uint64) ([]uint64, []FP) {
	table := NewHashTable(n, numBits)
	table.AddAll(h, data)
	return convertAndCap(table.hashes)
}

func (t *HashTable) AddAll(h hash.Hash, data []*vec.Vec) {
	numThreads := runtime.NumCPU()
	sections := hash.Spans(len(data), numThreads)
	errs := make(chan error)
	for i := 0; i < numThreads; i++ {
		go func(i int) {
			myHashes := make(map[uint64][]uint32)
			for row := sections[i][0]; row < sections[i][1]; row++ {
				v := data[row]
				hash := h.Hash(v)
				hash = hash & t.mask
				cur := myHashes[hash]
				myHashes[hash] = append(cur, uint32(row))
				// to give some sense of progress
				if (row & 16383) == 10000 {
					log.Printf("[Server]: table %d, completed row %v of %v\n", t.table, row-sections[i][0], sections[i][1]-sections[i][0])
				}
			}
			t.mu.Lock()
			t.Merge(myHashes)
			t.mu.Unlock()
			errs <- nil
		}(i)
	}
	for i := 0; i < numThreads; i++ {
		<-errs
	}
}

func (t *HashTable) Merge(other map[uint64][]uint32) {
	for k, v := range other {
		cur := t.hashes[k]
		if len(cur) > 0 {
			t.hashes[k] = append(t.hashes[k], v...)
		} else {
			t.hashes[k] = v
		}
	}
}

func (t *HashTable) Len() int {
	return len(t.hashes)
}

// Choose one element to keep from each bucket with multiple values
func convertAndCap(hashTable map[uint64][]uint32) ([]uint64, []FP) {
	keys := make([]uint64, 0)
	values := make([]FP, 0)
	for k, v := range hashTable {
		r := 0
		if len(v) > 1 {
			r = rand.Intn(len(v))
		}
		keys = append(keys, k)
		values = append(values, FP(v[r]))
	}
	return keys, values
}

func (t *HashTable) Get(h uint64) []uint32 {
	h = h & t.mask
	return t.hashes[h]
}
