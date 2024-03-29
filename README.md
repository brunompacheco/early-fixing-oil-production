# Supervised and Weakly-supervised Deep-learning-based Early Fixing

This repository accompanies the paper "Deep-learning-based Early Fixing for Gas-lifted Oil Production Optimization: Supervised and Weakly-supervised Approaches" (accepted at SBAI'23). The abstract follows

> Maximizing oil production from gas-lifted oil wells entails solving Mixed-Integer Linear Programs (MILPs). As the parameters of the wells, such as the basic-sediment-to-water ratio and the gas-oil ratio, are updated, the problems must be solved repeatedly. Instead of relying on costly exact methods or the accuracy of approximate methods, in this paper, we train deep-learning-based heuristics that provide values for all integer variables based on the parameters of the wells, early fixing their values in the original problem, which becomes a linear program. We propose two approaches for developing such a heuristic: a supervised learning approach, which requires the optimal integer values for several instances of the original problem to train a deep learning model, and a weakly-supervised learning approach, which requires only solutions for the early-fixed linear problem with random integer values. Our results show that our early-fixing heuristics reduce the runtime in 82.11\%, and that the supervised learning model can correctly guess the optimal integer values 99.78\% of the time. Furthermore, the weakly-supervised learning model can provide significant values for early fixing, despite never seeing the optimal values during training.

The code is developed using my boilerplate code for deep learning projects, using PyTorch and tracking experiments through W&B.
The notebooks (`notebooks/`) were used for exploring the data, generating visualizations and testing.
The important stuff is all in `src/`.
