# Efficient-HRL-MetaLearning

This repository provides a reference implementation of **Hierarchical Reinforcement Learning (HRL) with Model-Agnostic Meta-Learning (MAML)**, as described in the thesis:

> **Efficient Adaptation in Reinforcement Learning via Hierarchical Meta-Learning**  
> [Victor V. S.](https://github.com/vvsvictor)

The code demonstrates how to combine hierarchical policies (FeUdal Networks-style) and meta-learning (MAML) to achieve fast adaptation and sample efficiency in challenging RL environments.

---

## Features

- **Hierarchical RL**: FeUdal Networks architecture (Manager/Worker) for temporal abstraction.
- **Meta-Learning**: Model-Agnostic Meta-Learning (MAML) for few-shot adaptation.
- **Lightweight Environments**: Tested on [MiniGrid](https://minigrid.farama.org/) for reproducibility on modest hardware.
- **Baselines Included**: Standard Actor-Critic and pure HRL for comparison.
- **Reproducible experiments**: Scripts for training, evaluation, and logging.
