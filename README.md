# GCN-Solubility-Prediction

A molecular solubility prediction project based on Graph Convolutional Networks (GCN).

## Overview

This project uses a Graph Convolutional Network (GCN) to predict the aqueous solubility (LogS) of molecules from their molecular structures.

Molecular solubility is an important property in drug discovery, as it can affect drug absorption and bioavailability. Traditional machine learning methods often rely on manually designed molecular descriptors or fingerprints. In this project, I wanted to explore whether a graph neural network could learn useful molecular representations directly from molecular structures.

The basic idea is to represent each molecule as a graph, where atoms are nodes and chemical bonds are edges. A GCN is then used to learn structural features, which are passed to a regression model to predict LogS.

## Method

The overall workflow consists of three main steps.

**1. Molecular representation**

Molecules are converted into graph structures based on their SMILES representations. Each graph contains atom features and information about the connections between atoms.

**2. Graph convolution**

The GCN updates atom representations by aggregating information from neighboring atoms. After several graph convolution operations, the model can capture information about the local chemical environment of each atom.

A graph pooling operation is used to combine the atom-level representations into a single molecular representation.

**3. Solubility prediction**

The molecular representation is passed through a regression layer to predict the LogS value.

The model is trained by comparing predicted values with experimental solubility values.

## Project Structure

```text
GCN-Solubility-Prediction/
├── config/             # Configuration files
├── scripts/
│   └── run.py          # Main entry point
├── src/                # Source code
├── requirements.txt    # Dependencies
└── README.md
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/ChloeLincc/GCN-Solubility-Prediction.git
cd GCN-Solubility-Prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python scripts/run.py
```

Make sure the dataset and configuration paths are set correctly before running.

## Experiments

The model is designed for a regression task, where the goal is to predict molecular solubility as accurately as possible.

Model performance can be evaluated using metrics such as RMSE, MAE, and R².

**TODO:** Add dataset details, training settings, and experimental results.

## Future Work

There are several things I would like to improve or explore further:

- Compare GCN with other graph neural network architectures, such as GAT and GIN.
- Experiment with different atom features and graph pooling methods.
- Evaluate model generalization using different dataset splitting strategies.
- Add visualizations of predicted and experimental LogS values.

## References

- Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.*
- Wu, Z., et al. (2018). *MoleculeNet: A Benchmark for Molecular Machine Learning.*
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)

## Author

[ChloeLincc](https://github.com/ChloeLincc)
