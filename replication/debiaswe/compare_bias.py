from __future__ import print_function, division
import debiaswe.we as we
import json
import numpy as np
import argparse
import sys
from matplotlib import pyplot as plt
from .data import load_professions, load_definitional_pairs
from .we import doPCA
if sys.version_info[0] < 3:
    import io
    open = io.open
"""
Example script for comparing occupational bias across word embeddings.
Following approach from:

Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""

def plot_comparison_embeddings(datapoints_a, datapoints_b, embedding_names):
    """
    Plot the occupational bias comparison across two embeddings.


    :param object E_a: WordEmbedding object.
    :param object E_b: WordEmbedding object.
    :returns: None
    """
    fig, ax = plt.subplots()
    ax.scatter(datapoints_b, datapoints_a, s=10)
    ax.set_ylim(-0.3, 0.5)
    ax.set_xlim(-0.3, 0.5)
    plt.xlabel("Gender axis of {}".format(embedding_names[1]), fontsize=11)
    plt.ylabel("Gender axis of {}".format(embedding_names[0]), fontsize=11)
    plt.title("Occupational gender bias across embeddings", pad=18, fontsize=13)
    fig.savefig("compare_bias.png")
    plt.show()


def get_datapoints_embedding(E, v_gender, professions, unique_occupations):
    """
    Get datapoints for one embedding.


    :param object E: WordEmbedding object.
    :param object list projection: List with projection of profession words
        onto gender axis of embedding.
    :param list unique_occupations: List of occupations present in all 
        embeddings to compare.
    :returns: datapoints list
    """

    # Extract datapoint per occupation and sort datapoints
    sp = sorted([(E.v(w).dot(v_gender), w) for w in professions if w in unique_occupations])
    points = [s[0] for s in sp]
    words = [s[1] for s in sp]
    words_sorted_ind = sorted(range(len(words)), key=lambda k: words[k])
    datapoints =  [points[i] for i in words_sorted_ind]
    return datapoints


def project_profession_words(E, professions, unique_words):
    """
    Get gender axis and project profession words onto this axis.

    :param object E: WordEmbedding object.
    :param list unique_words: List of words present in all 
        embeddings to compare.
    :returns: projection, profession words
    """
    # Extract definitional word embeddings and determine gender direction.
    defs = load_definitional_pairs()
   
    # TODO!!! 
    defs = [d for d in defs if d[0] in E.words and d[1] in E.words]

    v_gender = doPCA(defs, E).components_[0]
    # v_gender = E.diff('she', 'he')

    # Projection on the gender direction.
    sp = E.profession_stereotypes(professions, v_gender)

    occupations = [s[1] for s in sp]
    return sp, occupations, v_gender


def compare_occupational_bias(E_a, E_b, embedding_names):
    """
    Compare occupational bias across word embeddings.

    :param object E_a: WordEmbedding object.
    :param object E_b: WordEmbedding object.
    :param object list embedding_names: List with strings of names of 
        embeddings. For example, ["word2vec", "GloVe"].
    :returns: None
    """
    unique_words = list(set(E_a.words).intersection(E_b.words))
    professions = load_professions()
    proj_a, occupations_a, v_gender_a = project_profession_words(E_a, professions, unique_words)
    proj_b, occupations_b, v_gender_b = project_profession_words(E_b, professions, unique_words)
    unique_occupations = list(set(occupations_a).intersection(occupations_b))
    datapoints_a = get_datapoints_embedding(E_a, v_gender_a, professions, unique_occupations)
    datapoints_b = get_datapoints_embedding(E_b, v_gender_b, professions, unique_occupations)
    plot_comparison_embeddings(datapoints_a, datapoints_b, embedding_names)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("embedding_filename_a", help="The name of the embedding")
    parser.add_argument("embedding_filename_b", 
        help="The name of the embedding to compare with")
    parser.add_argument("embedding_names", 
        help="List of two strings with embedding names")

    if len(sys.argv[2] != 2):
        print("Please give third argument the names of two embeddings as list")

    args = parser.parse_args()
    print(args)

    compare_occupational_bias(embedding_filename_a, embedding_filename_b, 
        embedding_names)