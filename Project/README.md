<h1 align="center">Navier Stokes Spectral Optimization</h1>

<p align="center">
    <img src="https://github.com/Deadlica/DD2358/blob/main/Project/assets/navier-stokes-spectral.png" alt="Navier Stokes Spectral.png" width="400" height="400">
</p>


<br />

![pull](https://img.shields.io/github/issues-pr/deadlica/DD2358)
![issues](https://img.shields.io/github/issues/deadlica/DD2358)
![coverage](https://img.shields.io/codecov/c/github/deadlica/DD2358)
![language](https://img.shields.io/github/languages/top/deadlica/DD2358)

## Introduction
This repository provides two CPU-based, two GPU-based optimizations for Philip Mocz's simulation of the Navier-Stokes equations, the movement of fluid, specifically for an incompressible viscous fluid, using a spectral method. These optimizations are `Cython`, a python module allowing integration compiled C-like code to speed up python programs, `Algorithmic`, an optmization on the source code which adresses some of the flaws noted by the original author, `Cupy`, a library for GPU-accelerated computing, and `Pytorch`, a package which provides Tensor computation also with GPU-accelerated computing. Philips Mocz's repository can be found [here](https://github.com/pmocz/navier-stokes-spectral-python).

The program `execution_perf.py` plots the performance differences of the optimizations. These analyses show the execution time of the various implementations [Cupy, Pytorch, Algorithmic and non-optimized], whilst varying the values of certain parameters, those parameters being: **N**, the spatial resolution, **dt**, the time-step.

There is also a second program `compare_execution_perf.py` which plots all of the optimization techiniques exeuction time for a varied parameter in the same graph.

## Dependencies
The following technologies have been tested and are required to run the project:
- Python 3.11.7 or newer
- Pip 24.0 or newer
- GNU Make 4.4.1 (Only neccesary if you want to run with make commands)
- Nvidia CUDA GPU (The GPU benchmarks won't work otherwise)

`Note: older version might work however, they have not been tested.`

Once you have the necessary dependencies you can install all the required python modules with the following make command:
```bash
make requirements
```
or without make
```bash
pip install -r requirements.txt
```

## Documentation
Extensive documentation can be found [here](https://deadlica.github.io/DD2358/).

If you would like to generate new documentation pages make sure to install the required theme:
```bash
pip install sphinx rtd_sphinx_theme
```
and thereafter simply run:
```bash
make docs
```
from the root directory.

`Note: this step requires Make since Sphinx themselves use make commands to generate the HTML pages`

## Tests
There are unit tests written for the GPU optimization to ensure that the computations match the original solution. The following make commands will run these tests.

With Make:
```bash
make test
```

Without Make:
```bash
pytest Unit_Tests/test_cupy.py && pytest Unit_Tests/test_pytorch.py
```


## Usage (execution_perf.py)
There are multiple ways in which this project can be run. The program supports multiple CLI arguments that both affect what underlying module is used to compute the Navier Stokes and choosing what parameters to vary.

The default setting (no CLI arguments) for the code will vary the variables **N** (spatial resolution), **dt** (timestep) and the underlying module will be `Pytorch`. To run this configuration use the following command:
```bash
python3 execution_perf.py
```

<a id="choose-module"></a>
### Choosing module for computation
There are four options to choose from. Pytorch (GPU), Cupy (GPU), Algorithmic (CPU) and Numpy (CPU). The following commands show you have to run each of the modules respectively.

#### Pytorch
* With Make
   ```bash
    make pytorch
    ```

* Without Make
   ```bash
    python3 execution_perf.py  pytorch
    ```

#### Cupy
* With Make
   ```bash
    make cupy
    ```

* Without Make
   ```bash
    python3 execution_perf.py  cupy
    ```

#### Numpy
* With Make
   ```bash
    make numpy
    ```

* Without Make
   ```bash
    python3 execution_perf.py  numpy
    ```

#### Algorithmic
* With Make
   ```bash
    make algo
    ```

* Without Make
   ```bash
    python3 execution_perf.py  algo
    ```

### Choosing what parameter to vary
There is also the possibility to choose what parameters for the Navier Stokes to be varied when measuring the execution time. By default all parameters are varied as mentioned in the previous [section](#choose-module). When running the program with this method the `Pytorch` module will always be used.

The syntax for specifying which parameters are to varied is as follows:
***
`param1`, `param2`: These arguments are optional. They can each take one of the following values: `res`, `time`.
```bash
python3 execution_perf.py [param1] [param2]
```

Here's an example of how you can run the program with different combinations of arguments:

```bash
python3 execution_perf.py res
python3 execution_perf.py time res
```

There are some make commands provided for varying `res`, `time` and individually.

```bash
make res
make time
```

## Usage (compare_execution_perf.py)
The purpose of `compare_execution_perf.py` is to compare the execution time of all the optimization techniques for a varied parameter in the same graph. This also makes the CLI argument usage simpler.

The default usage of this program will plot two graphs, varying N, dt respectively where each graph shows the execution time for each of the optimizations per graph.

With Make:
```bash
make comp
```

Without Make:
```bash
python3 compare_execution_perf.py
```

There is also the possibility to choose what parameters to vary as with `execution_perf.py`.

The syntax for specifying which parameters are to varied is as follows:
***
`param1`, `param2`: These arguments are optional. They can each take one of the following values: `res`, `time`.
```bash
python3 compare_execution_perf.py [param1] [param2]
```

Here's an example of how you can run the program with different combinations of arguments:

```bash
python3 compare_execution_perf.py res
python3 compare_execution_perf.py time res
```

There are some make commands provided for varying `res`, `time` and individually.

```bash
make comp_res
make comp_time
```
