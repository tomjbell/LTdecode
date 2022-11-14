# LTdecode
Repository for code to analyse the loss tolerant properties of graph codes for measurement-based quantum error correction.

The contents of this repository allow for the analysis of small graph codes, encoded using progenitor graph states of up to 12 qubits.
Functions are provided to build decision-tree decoders to determine which qubit of the graph code to measure and in which basis, to perform a logical single qubit or fusion measurement of the graph codes.

A walk through of the different functionalities is described in greater detail in the jupyter notebook `decoding_example.ipynb`

The following packages are required for basic functionality:
* numpy
* matplotlib
* networkx

