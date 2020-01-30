# Embetter: FactAI Word Embedding Debiassing 

Are you using pre-trained word embeddings like word2vec, GloVe or fastText? `Embetter` allows you to easily remove gender bias.
No longer can nurses only be female or are males the only cowards.

This repository was made during the FACT-AI course at the University of Amsterdam, during which papers from the FACT field are reproduced and possibly extended.

## Getting Started

Here we will outline how to access the code.

### Prerequisites

To run the code, create an Anaconda environment using:
```
conda env create -f environment.yml
```
or create an empty environment, install pip in this environment and run:
```
pip install -r requirements.txt
```

This will install all dependencies for this package.

### Installing

To use the code from this package, simply download or clone the repository.

## Available pre-trained embeddings
Several pre-trained embeddings are provided and are automatically downloaded upon creating a `WordEmbedding` object with the name of one of the available embeddings. The embeddings are based on <a href="https://code.google.com/archive/p/word2vec/">Word2vec</a>, <a href="https://nlp.stanford.edu/projects/glove/">GloVe</a>, and <a href="https://fasttext.cc/">fastText</a>.

The available embeddings are listed below.

| Embedding name | dimensionality | vocabulary size | 
| ------------- |:-------------:| -----|
| `word2vec_large`     | 300 | 3M |
| `word2vec_small`     | 300 | 26423 |
| `word2vec_small_hard_debiased`     | 300 | 26423 |
| `word2vec_small_soft_debiased`     | 300 | 26423 |
| `glove_large`     | 300 | 1.9M |
| `glove_small`     | 300 | 42982 |
| `glove_small_hard_debiased`     | 300 | 42982 |
| `glove_small_soft_debiased`     | 300 | 42982 |
| `fasttext_large`     | 300 | 999994 |
| `fasttext_small`     | 300 | 27014 |
| `fasttext_small_hard_debiased`     | 300 | 27014 |
| `fasttext_small_soft_debiased`     | 300 | 27014 |

Note that because of the large computational workload, hard debiasing of large embeddings (entire vocabulary) is very difficult and soft debiasing with the current setup is impossible, and are therefore not available for download.

### Embedding from file
Besides the available embeddings, it is also possible to load embeddings from a file on your device. If the embedding is very large and loading the embedding takes unreasonably long, make sure the filename ends with "large" (possibly preceding a file extension). E.g. `./myfolder/myembedding_large.txt`. This makes sure that the embedding is loaded using the `gensim` package, which scales better with large embeddings. However, if you do this, the first line of the file **must** contain the number of words in the embedding, followed by a space, followed by the embedding dimensionality, e.g. for `glove_large` this is `1900000 300`.

Besides this first line for large embeddings, each line in the file must start with a word from the vocabulary, followed by the vector values, separated by whitespace characters. Binary (.bin) files are also supported for the available word2vec binary embedding files, but support is not guaranteed for manually created binary files and should not be heavily relied upon.

#### Example
```
from embetter.we import WordEmbedding

w2v = WordEmbedding('word2vec_small')

my_embed = WordEmbedding('./myfolder/myembedding.txt')
```

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

The debiasing methods that we provide have been proposed by 
> Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings \
> Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai \
> 2016 (https://arxiv.org/abs/1607.06520)

and full credit for these methods goes to them.

The code that we provide is largely based on their code, which is available in their Github repository: \
https://github.com/tolga-b/debiaswe.

Additionally, some of the code borrows from:
- https://github.com/k-kawakami/embedding-evaluation
- https://github.com/chadaeun/weat_replication
- https://stackoverflow.com/a/39225272
