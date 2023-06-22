# Quantum Gate Simulation on GPU
This repository contains code that performs simulation of applying a
single-qubit quantum gate to one qubit in an n-qubit quantum circuit. 

There are two versions of the algorithm: `quamsimV1.cu` uses `cudaMalloc`
while the second version uses `cudaMallocManaged` to see improvement
otherwise identical code.
 
## Reference
1. HyQuas: Hybrid Partitioner Based Quantum Circuit Simulation System on GPU

## How to Run
1. Use the `setup.py` file to prepare the enviroment (`python setup.py`)
2. Run `make` to build the executables (`quamsimV1` and `quamsimV2`)
3. Use the following format to run a simulation:
```
./quamsimV1 input.txt
```
