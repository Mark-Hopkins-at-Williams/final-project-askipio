# CSCI 381 Final Project

Group members:
- Alex Kim


This is the searchless chess project, inspired by [this recent paper from Google DeepMind.](http://arxiv.org/pdf/2402.04494)


A key bottleneck for actually training a *good* chess playing transformers is that we need scale. Millions of parameters at a minimum, which I haven't been able to run on my machine due to hardware constraints. 

The src file contains the source code for the searchless_chess.src library written by Google DeepMind researchers. Inspired by their work, I present *compare_policies.ipynb* as the executable for this project. It is designed to run in a standard jupyter env. It contains code that examines three different policies 'action value', 'behavioral cloning', and 'state value'. It also compares the 'action value' policy at three differnet engine temperatures.  

In this repo is also the pdf of the original paper and my presentation slides. 

Requirements (may not need all):

absl-py
apache-beam
chess
chex
dm-haiku
grain-nightly
jax
jaxtyping
jupyter
numpy
optax
orbax-checkpoint
pandas
scipy
typing-extensions



