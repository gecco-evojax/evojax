# EvoJAX: Hardware-Accelerated Evolution Strategies

This repository contains the implementation of EvoJAX for the GECCO 2022 submission.

## Code Overview

EvoJAX has 3 major components, which we expect the users to extend.
1. **Evolution Strategies** All ES algorithms should implement the
`evojax.algo.base.ESAlgorithm` interface and reside in `evojax/algo/`.
We current provide [PGPE](https://people.idsia.ch/~juergen/nn2010.pdf), with more coming soon.
2. **Policy Networks** All neural networks should implement the
`evojax.policy.base.PolicyNetwork` interface and be saved in `evojax/policy/`.
In this repo, we give example implementations of the MLP, Convnet and Seq2Seq models.
3. **Tasks** All tasks should implement `evojax.task.base.VectorizedTask`
and be in `evojax/task/`. We highlight 6 non-trivial demo tasks in this codebase (see details below).

In addition, `evojax.trainer` and `evojax.sim_mgr` manage the
training pipeline. Although we plan to improve the implementations, they should
be sufficient for the current policies and tasks.

## Examples

As a quickstart, we provide examples in the `notebooks/` folder:

*Supervised Learning Tasks*
* [MNIST Classification](https://github.com/gecco-evojax/evojax/blob/main/notebooks/MNIST.ipynb) - 
we show that EvoJAX trains a Convnet policy to achieve >98% test accuracy within 5 min on a single GPU.
* [Seq2Seq Learning](https://github.com/gecco-evojax/evojax/blob/main/notebooks/Seq2SeqTask.ipynb) -
We demonstrate that EvoJAX is capable of learning a large
network with hundreds of thousands parameters to accomplish a seq2seq task.

*Classic Control Tasks*
* [Locomotion](https://github.com/gecco-evojax/evojax/blob/main/notebooks/BraxTasks.ipynb) -
[Brax](https://github.com/google/brax) is a differentiable physics
engine implemented in JAX. We wrap it as a task and train with
EvoJAX on GPUs/TPUs. It takes EvoJAX tens of minutes to solve a locomotion task in Brax.
* [Cart-Pole Swing Up](https://github.com/gecco-evojax/evojax/blob/main/notebooks/CartPole.ipynb) -
We illustrate how the classic control task can be implemented in JAX and be
integrated into EvoJAX' pipeline for significant speed up training.

*Novel Tasks*
* [WaterWorld](https://github.com/gecco-evojax/evojax/blob/main/notebooks/WaterWorld.ipynb) -
In this [task](https://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html), an
agent tries to get as much food as possible while avoiding poisons. EvoJAX is
able to learn the agent in tens of minutes on a single GPU. Moreover, we
demonstrate that [multi-agents training](https://github.com/gecco-evojax/evojax/blob/main/notebooks/MultiAgentsWaterWorld.ipynb)
in EvoJAX is possible, which is beneficial for learning policies that can deal with
environmental complexity and uncertainties.
* [Abstract](https://github.com/gecco-evojax/evojax/blob/main/notebooks/AbstractPainting01.ipynb) [Painting](https://github.com/gecco-evojax/evojax/blob/main/notebooks/AbstractPainting02.ipynb) - We reproduce the results from this [computational creativity
work](https://es-clip.github.io/) and show how the original work, whose
implementation requires multiple CPUs and GPUs, could be accelerated on a single
GPU efficiently using EvoJAX, which was not possible before. Moreover, with multiple
GPUs/TPUs, EvoJAX can further speed up the mentioned work almost linearly.
We also show that the modular design of EvoJAX allows its components
be used independently -- in this case it is possible to use only the ES algorithms
from EvoJAX while leveraging one's own training loops and environment implantation.
