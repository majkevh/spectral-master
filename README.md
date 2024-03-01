# LITE: A Framework for Lattice-Integrated Embedding of Topological Descriptors
**Author**: Michael Etienne Van Huffel

## Description
Implementation of Lattice Integrated Topological Embedding (LITE) for Persistence Diagrams based on the paper https://arxiv.org/abs/2312.17093.

## Overview
This repository contains the official implementation of LITE, a method for embedding Persistence Diagrams into elements of vetor spaces. It showcases the integration of such vectorization technique in the context of graph classification and dynamical particles classification.

## Repository Structure
The repository is systematically organized to facilitate easy navigation and comprehensive understanding of each component.

### Jupyter Notebooks
- `graph-classy.ipynb`: Demo that demonstrates the application of LITE Vectorization in graph classification.
- `orbit-classy.ipynb`: Demo that demonstrates the application of LITE Vectorization in dynamical particles classification.

### Python Scripts
- `lite/lite.py`: Core implementation of the LITE Vectorization algorithm.
- `lite/utils.py`: Provides essential utility functions for data processing and analysis within the notebooks.

## Installation
To reproduce the analysis environment, you will need Python 3.6 or later. Please install the required Python packages listed in `requirements.txt`.

```bash
git clone git@github.com:majkevh/spectral-master.git
cd spectral-master
pip install -r requirements.txt
```

### Data Folder 
- `data/`: Contains all datasets used in the notebooks. For conducting experiments involving graph data, I utilized datasets and functions sourced from the [*PersLay*](https://github.com/MathieuCarriere/perslay) and [*ATOL*](https://github.com/martinroyer/atol) repositories.


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
