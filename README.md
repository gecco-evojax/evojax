# EvoJAX: Hardware-Accelerated Evolution Strategies

This repository contains the implementation of EvoJAX for the GECCO 2022 submission.

## Code Overview

EvoJAX has 3 major components, which we expect the users to extend.
1. **Evolution Strategies** All ES algorithms should implement the
`evojax.algo.base.ESAlgorithm` interface and reside in `evojax/algo/`.
2. **Policy Networks** All neural networks should implement the
`evojax.policy.base.PolicyNetwork` interface reside in `evojax/policy/`.
3. **Tasks** All tasks should implement `evojax.task.base.VectorizedTask`
and in `evojax/task/`.

In addition, `evojax.trainer` and `evojax.sim_mgr` manage the
training pipeline. Although we plan to improve the implementations, they should
be sufficient for the current policies and tasks.

## Examples

As a quickstart, we provide examples in the `notebook/` folder:

*Supervised Learning Tasks*
1. MNIST Classification - We show that EvoJAX trains a Convnet policy to
achieve >98% test accuracy within 5 min on a single GPU.
2. Seq2Seq Learning - We demonstrate that EvoJAX is capable of learning a large
network with hundreds of thousands parameters to accomplish a seq2seq
[task](https://github.com/google/flax/tree/main/examples/seq2seq).

*Classic Control Tasks*
3. Locomotion - [Brax](https://github.com/google/brax) is a differentiable physics
engine implemented in JAX. We wrap it as a task and train with
EvoJAX on GPUs/TPUs. It takes EvoJAX tens of minutes to solve a locomotion task
in Brax.
4. Cart-Pole Swing Up - We illustrate how the classic control task can be
implemented in JAX and be integrated into EvoJAX' pipeline for significant
speed up training.

*Novel Tasks*
5. WaterWorld - In this
[task](https://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html), an
agent tries to get as much food as possible while avoiding poisons. EvoJAX is
able to learn the agent in tens of minutes on a single GPU. Moreover, we
demonstrate that multi-agents training in EvoJAX is possible, which is
beneficial for learning policies that can deal with environmental complexity and
uncertainties.
6. Abstract Painting - We reproduce the results from
[this](https://es-clip.github.io/) art work and show how to accelerate the
original work on a single GPU, which was not possible before. With multiple
GPUs/TPUs, EvoJAX can further speed up the training. In this example, we show
that EvoJAX has independent components. It is possible to use only the
ES algorithms from EvoJAX and write one's own training loops.
