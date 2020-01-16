import numpy as np
import os
DATA_ROOT = os.path.dirname( os.path.abspath( __file__ ) ) + "/benchmarks/"

def MSR(E, discount_query_words=False):
    """
    Executes MSR-analogy benchmark on the word embeddings in E


    :param object E: WordEmbedding object.
    :param boolean discount_query_words: Give analogy solutions that appear in
        the query 0 score.
    :returns: Percentage of correct analogies (accuracy), number of OOV words
    """
    # Load and format the benchmark data
    analogy_answers = np.genfromtxt(
        DATA_ROOT + "word_relationship.answers",
        dtype='str', encoding='utf-8')
    analogy_a = np.expand_dims(analogy_answers[:,1], axis=1)
    analogy_q = np.genfromtxt(DATA_ROOT + "word_relationship.questions",
        dtype='str', encoding='utf-8')

    # Remove Out Of Vocabulary words_not_found
    analogy_stack = np.hstack((analogy_a, analogy_q))
    present_words = np.isin(analogy_stack, E.words).all(axis=1)
    filtered_answers = analogy_a[present_words]
    filtered_questions = analogy_q[present_words]

    # Extract relevant embeddings from E
    a = E.vecs[np.vectorize(E.index.__getitem__)(filtered_questions[:,0])]
    x = E.vecs[np.vectorize(E.index.__getitem__)(filtered_questions[:,1])]
    b = E.vecs[np.vectorize(E.index.__getitem__)(filtered_questions[:,2])]
    all_y = E.vecs

    # Calculate scores
    y_pos = ((1+all_y@x.T)/2)*((1+all_y@b.T)/2)
    y_neg = (1+all_y@a.T+0.00000001)/2
    y_scores = y_pos/y_neg

    # If set, set scores of query words to 0
    if discount_query_words:
        query_word_ind = np.vectorize(E.index.__getitem__)(filtered_questions).T
        y_scores[query_word_ind, np.arange(y_scores.shape[1])[None,:]] = 0

    # Retrieve words with best analogy scores
    y = np.expand_dims(np.array(E.words)[np.argmax(y_scores, axis=0)], axis=1)

    # Calculate returnable metrics
    accuracy = np.mean(y==filtered_answers)*100
    words_not_found = len(analogy_answers) - len(filtered_answers)

    return accuracy, words_not_found
