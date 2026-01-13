# research_placement_spectral_methods
This repository contains the current code for the active learning framework to accelarate spectral energy simulation for the multiscale analysis of neo-hookian metamaterials.

Purpose:
This framework aims to use machine learning to accelarate the optimisation steps during the energy minimisation process used in non-linear response analysis, this research is currently ongoing so this code is not final and is still unfinished.

Features:
Currently the framework uses Mahalanobis filtering to trigger simulator query. A gated resnet is used to predict strain energy from strain polynomial coefficients. We chose a spectral approach having tested an approach using FEA and GNNs and finding that spectral methods in this domain offer far more efficient computation and simplify the prediction job of the ML algorithm by providing a continuous displacement function oppose to a set of nodes. 