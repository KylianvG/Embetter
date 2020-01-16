import os
import numpy as np
from collections import defaultdict
from scipy import linalg, mat, dot, stats
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


class Wordsim:
    def __init__(self):
        self.files = [ file_name.replace(".txt","") for file_name in os.listdir(DATA_ROOT) if ".txt" in file_name ]
        self.dataset=defaultdict(list)
        for file_name in self.files:
            for line in open(DATA_ROOT + "/" + file_name + ".txt"):
                self.dataset[file_name].append([ float(w) if i == 2 else w for i, w in enumerate(line.strip().split())])

    @staticmethod
    def cos(vec1,vec2):
        return vec1.dot(vec2)/(linalg.norm(vec1)*linalg.norm(vec2))

    @staticmethod
    def rho(vec1,vec2):
        return stats.stats.spearmanr(vec1, vec2)[0]

    @staticmethod
    def pprint(result, method):
        from prettytable import PrettyTable
        table = PrettyTable(["Dataset", "Found", "Not Found", "Score (rho)"])
        table.title = 'Results for {}'.format(method)
        table.align["Dataset"] = "l"
        for k, v in result.items():
            table.add_row([k,v[0],v[1],v[2]])
        print(table)

    @staticmethod
    def pprint_w2vnews(results, methods, title):
        assert len(results) == len(methods)
        from prettytable import PrettyTable
        table = PrettyTable(["Score (rho)", "EN-RG-65", "EN-WS-353-ALL"])
        table.title = 'Results for {} dataset'.format(title)
        for result, method in zip(results, methods):
            table.add_row([method, list(result.values())[0][2], list(result.values())[1][2]])
        print(table)

    def evaluate(self, word_dict, method, print=True):
        result = {}
        vocab = word_dict.keys()
        for file_name, data in self.dataset.items():
            pred, label, found, notfound = [] ,[], 0, 0
            for datum in data:
                if datum[0] in vocab and datum[1] in vocab:
                    found += 1
                    pred.append(self.cos(word_dict[datum[0]],word_dict[datum[1]]))
                    label.append(datum[2])
                else:
                    notfound += 1
            result[file_name] = [found, notfound, self.rho(label,pred)*100]
        if print:
            self.pprint(result, method)
        return result