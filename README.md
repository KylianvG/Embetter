# FactAI Word Embedding Debiassing

Using pre-trained word embeddings like w2v, GloVe or fastText? This code allows you to easily remove any societal bias.
No longer can nurses only be female or are males the only cowards.

This repository is used for the FACT-AI course, where we replicate the experiments of a paper from the FACT field.

## Getting Started

Here we'll outline how to access the code.

### Prerequisites

To run the code, create an Anaconda environment using:
```
conda env create -f environment.yml
```
or create an empty environment, install pip in this environment and run:\
```
pip install -r requirements.txt
```

This will install all dependencies for this package.

### Installing

To use the code from this package, simply download or clone the repository.

## Running the experiments

A general tutorial on the usage of this package, together with some of the experiments are available in the Jupyter notebook in the repository.\
For a full replication of all the available experiments, you can execute the `experiments.py` script.\
Use
```
python experiments.py -h
```
in your terminal to get a list of options on which experiments to run.

## Authors

Kylian van Geijtenbeek -11226145 - kylian.vangeijtenbeek@gmail.com \
Thom Visser - 11281405 - thom.1@kpnmail.nl \
Martine Toering - 11302925 - martine.toering@student.uva.nl \
Iulia Ionescu - 10431357 - iulia.ionescu@student.uva.nl

Morris Frank - TA

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgements

This code is extended from the code from:

> Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings \
> Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai \
> 2016

Which is available at: \
https://github.com/tolga-b/debiaswe

Incorporating other extentions from:

- https://github.com/k-kawakami/embedding-evaluation
- https://github.com/chadaeun/weat_replication
- https://stackoverflow.com/a/39225272
