import json
import os

"""
Tools for data operations.
Extended from the code from:

Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""
PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def load_professions(load_scores=False, available_words=[]):
    """
    Loads professions from data file

    :param bool load_scores: Load bias scores if True. Defaults to False.
    :returns: List of professions
    """
    professions_file = os.path.join(PKG_DIR, '../data', 'professions.json')
    with open(professions_file, 'r') as f:
        professions = json.load(f)
    # print('Loaded professions\n' +
          # 'Format:\n' +
          # 'word,\n' +
          # 'definitional female -1.0 -> definitional male 1.0\n' +
          # 'stereotypical female -1.0 -> stereotypical male 1.0')
    if not load_scores:
        professions = [p[0] for p in professions]
    return professions


def load_gender_seed():
    """
    Loads gender seed words from data file


    :returns: List of gender-specific words
    """
    gender_file = os.path.join(PKG_DIR, '../data', 'gender_specific_seed.json')
    with open(gender_file, 'r') as f:
        gender_words = json.load(f)
    return gender_words


def load_equalize_pairs():
    """
    Loads equalize pairs from data file


    :returns: List of equalize pairs
    """
    eq_file = os.path.join(PKG_DIR, '../data', 'equalize_pairs.json')
    with open(eq_file, 'r') as f:
        eq_pairs = json.load(f)
    return eq_pairs


def load_definitional_pairs():
    """
    Loads definitional pairs from data file


    :returns: List of definitional pairs
    """
    def_file = os.path.join(PKG_DIR, '../data', 'definitional_pairs.json')
    with open(def_file, 'r') as f:
        def_pairs = json.load(f)
    return def_pairs


def load_data():
    """
    Loads all data needed for debiasing and inspecting gender bias
    in proffesions.

    :returns: List of gender-specific words, list of definitional pairs,
        list of equalize pairs, list of professions
    """
    profs = load_professions()
    gender_seed = load_gender_seed()
    eq_pairs = load_equalize_pairs()
    def_pairs = load_definitional_pairs()
    return gender_seed, def_pairs, eq_pairs, profs
