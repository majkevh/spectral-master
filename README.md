# Spectral Persistence: Persistence Signals
**Author**: Michael Etienne Van Huffel

## Description
Implementation of Persistence Signals Vectorization for Persistence Measures based on the paper https://arxiv.org/abs/2312.17093.

## Overview
This repository contains the official implementation of Persistence Signals Vectorization, a method for embedding Persistence Measures in vetor spaces. It showcases the integration of such vectorization technique in the context of graph classification and dynamical particles classification.

## Repository Structure
The repository is systematically organized to facilitate easy navigation and comprehensive understanding of each component.

### Jupyter Notebooks
- `graph-classy.ipynb`: Demo that demonstrates the application of Persistence Signals Vectorization in graph classification.
- `orbit-classy.ipynb`: Demo that demonstrates the application of Persistence Signals Vectorization in dynamical particles classification.

### Python Scripts
- `spectral/signals.py`: Core implementation of the Persistence Signals Vectorization algorithm.
- `spectral/utils.py`: Provides essential utility functions for data processing and analysis within the notebooks.


### Documentation and Licensing
- `LICENSE`: Specifies the licensing terms for the use and distribution of the repository's content.

### Data Folder 
- `data/`: Contains all datasets used in the notebooks. For conducting experiments involving graph data, I utilized datasets and functions sourced from the [*PersLay*](https://github.com/MathieuCarriere/perslay) and [*ATOL*](https://github.com/martinroyer/atol) repositories.

