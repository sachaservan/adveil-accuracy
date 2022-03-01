# AdVeil accuracy experiments on real-world data

### Datasets 
All four datasets can be obtained from https://github.com/erikbern/ann-benchmarks.

The raw data is in HDF5 format which can be converted to CSV using ```/datasets/dataconv.py``` script. 
The python script generates three files prefixed by the dataset name. 
For example, ```python dataconv.py deep1b.hdf5``` will output *deep1b_train.csv*, *deep1b_test.csv*, and *deep1b_neighbors.csv*. 

The bash script argument requires ```DATASET_PATH``` point to the directory where these three files are located as well as the dataset name predix. 
For example, to run the server on the *deep1b* data, set```DATASET_PATH=/home/user/datasets/deep1b``` (note the lack of suffix in the dataset file name).
The code will automatically locate and use the training data to build the data structure and the test data as "queries" issued by clients. 
Note that generating the hash tables for the first time can take a while; we recommend caching the results. 

### Finding dataset parameters (Optional)
Note that all paramters are already pre-computed (located in ```/ann/cmd/meanAndStd/```).
However, follow the below steps if you would like to recompute or change the way the dataset parameters are generated. 

First go to the parameters directory 
```
cd ann/cmd/parameters
go build
```
To find the mean and standard deviation of the brute force distances for a dataset
```
./parameters --dataset ../../../datasets/mnist --dimension 24
```
The dimension 24 argument cause dimensionality reduction to 24 dimensions before calculating the distances,
so the radii will account for the variance introduced by the reduction.
This is the form expected by the implementation of the 24 dimensional Leech Lattice LSH.

### Checking hash function accuracy

First go to the accuracy directory
```
cd ann/cmd/accuracy
go build
```
The accuracy script accepts parameters for many aspects of the LSH.  For example
```
./accuracy --dataset=../../../datasets/mnist --mode=test --tables=2 --probes=5 --projectionwidthmean=887.77 --projectionwidthstddev=244.92
```
Evaluates the 10000 test queries for accuracy under approximation factor 2 for the MNIST dataset.
The values of width and stddev are those found with the parameter program.
To use training data to modify parameters, first run the parameter program to generate an answer set, move it into the directory, and use --mode=train.

The test.py python file contains the parameters used to run the experiments.


## Acknowledgements 
* The code is a modified version of the [Private ANN](https://github.com/sachaservan/private-ann) evaluation. 

## License
Copyright © 2022 Sacha Servan-Schreiber

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
