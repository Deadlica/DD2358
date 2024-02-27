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
This repository provides two GPU-based optimizations for Philip Mocz's simulation of the Navier-Stokes equations, the movement of fluid, specifically for an incompressible viscous fluid, using a spectral method. These optimizations are `Cupy`, a library for GPU-accelerated computing, and `Pytorch`, a package which provides Tensor computation also with GPU-accelerated computing. Philips Mocz's repository can be found [here](https://github.com/pmocz/navier-stokes-spectral-python).

The program `execution_perf.py` plots the performance differences of the optimizations. These analyses show the execution time of the various implementations [Cupy, Pytorch and non-optimized], whilst varying the values of certain parameters, those parameters being: **N**, the spatial resolution, **dt**, the
time-step, and **nu**, the viscosity.

## Dependencies
The following technologies have been tested and are required to run the project:
- Python 3.11.7 or newer
- Pip 24.0 or newer
- GNU Make 4.4.1 (Only neccesary if you to run with make commands)

`Note: older version might work but we have not tested these.`

Once you have the necessary dependencies you can install all the required python modules with the following make command:
```bash
$ make requirements
```
or without make
```bash
$ pip install -r requirements.txt
```

## Documentation
Extensive documentation can be found at https://deadlica.github.io/DD2358/

If you would like to generate new documentation pages make sure to install the required theme:
```bash
$ pip install sphinx rtd_sphinx_theme
```
and thereafter simply run:
```bash
$ make docs
```
from the root directory.

`Note: this step requires Make since Sphinx themselves use make commands to generate the HTML pages`

## Usage
There are multiple way in which the project can be run. The program supports multiple CLI arguments that both affect what underlying module is used to compute the Navier Stokes and choosing what which parameters to vary.

The default setting (no CLI arguments) for the code will vary the variables N (spatial resolution), dt (timestep) and nu (viscosity) and the underlying module will be Pytorch. To run this configuration use the following command:
```bash
$ python3 execution_perf.py
```

<a id="choose-module"></a>
### Choosing module for computation
There are three options to choose from. Pytorch (GPU), Cupy (GPU) and Numpy (CPU). The following commands show you have to run each of the modules respectively.

#### Pytorch
* With Make
   ```bash
    $ make pytorch
    ```

* Without Make
   ```bash
    $ python3 execution_perf.py  pytorch
    ```

#### Cupy
* With Make
   ```bash
    $ make cupy
    ```

* Without Make
   ```bash
    $ python3 execution_perf.py  cupy
    ```

#### Numpy
* With Make
   ```bash
    $ make numpy
    ```

* Without Make
   ```bash
    $ python3 execution_perf.py  numpy
    ```

### Choosing what parameter to vary
There is also the possibility to choose what parameters for the Navier Stokes to be varied when measuring the execution time. By default all three parameters are varied as mentioned in the previous [section](#choose-module). When running the program with this method the `Pytorch` module will always be used.

The syntax for specifying which parameters are to varied is as follows:
***
`param1`, `param2`, `param3`: These arguments are optional. They can each take one of the following values: `res`, `dt`, `nu`.
```bash
$ python3 execution_perf.py [param1] [param2] [param3]
```

Here's an example of how you can run the program with different combinations of arguments:

```bash
$ python3 run.py res
$ python3 run.py dt res
$ python3 run.py nu dt res
```

There are some make commands provided for varying `res`, `dt` and `nu`individually.

```bash
$ make res
$ make dt
$ make nu
```