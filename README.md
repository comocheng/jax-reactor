# JAX-reactor 
Simulating large chemical mechanisms is vital in combustion, atmospheric chemistry and heterogeneous catalysis. JAX-reactor is a package written in python to simulate large kinetic models leveraging just-in-time (JIT) compilation, automatic differentiaion and vectorization capabilities of awesome [JAX](https://github.com/google/jax) package. JAX uses [XLA](https://www.tensorflow.org/xla) to JIT compile python code to CPU, GPU and TPU. JAX can automatically differentiate python and numpy functions allowing to us efficiently calculate Jacobians of large chemical systems. 

JAX-reactor uses Cantera's recently developed YAML input format to read large detailed kinetic models. JAX-reactor is heavily inspired by a similar package written in PyTorch called [reactorch](https://github.com/DENG-MIT/reactorch). JAX-reactor is a research project and is in early stages of development. 

## Installation
First set up a conda environment

`conda create --name jax python=3.7 scons cython boost numpy=>1.5 ruamel_yaml scipy=>1.4  matplotlib jupyter` 

Install the dev build of Cantera 

`conda activate jax` 

`conda install -c cantera/label/dev cantera` 

CPU only version of JAX can be easily installed using `pip` 
```
pip install --upgrade pip
pip install --upgrade jax jaxlib  # CPU-only version
``` 

For GPU version of JAX please follow the official installation instructions at [JAX GPU installation](https://github.com/google/jax#pip-installation)

### Known JAX GPU build issue
If you want to use JAX on CentOS-7 you need too build JAX from source following instructions at https://github.com/google/jax/issues/2083 

Once the enviroment is setup install JAX-reactor:
```
git clone https://github.com/comocheng/jax-reactor.git
export PYTHONPATH=<full-path-to-cloned-folder>:$PYTHONPATH
```

## References
```
1. Weiqi Ji, Sili Deng. ReacTorch: A Differentiable Reacting Flow Simulation Package in PyTorch, https://github.com/DENG-MIT/reactorch, 2020.
```