import os
import numpy as np
from collections import defaultdict
from scipy import linalg, mat, dot, stats
import argparse
import debiaswe as we
DATA_ROOT = os.path.dirname( os.path.abspath( __file__ ) ) + "/benchmark_data/"

"""
Tools for benchmarking word embeddings.

Code adapted and extended from:
https://github.com/k-kawakami/embedding-evaluation

Using well-known benchmarks from:
(MSR)
    T. Mikolov, W.-t. Yih, and G. Zweig.
    Linguistic regularities in continuous space word representations.
    2013.
(RG)
    H. Rubenstein and J. B. Goodenough.
    Contextual correlates of synonymy.
    1965.
(WS)
    L. Finkelstein, E. Gabrilovich, Y. Matias, E. Rivlin, Z. Solan,
        G. Wolfman, and E. Ruppin.
    Placing search in context: The concept revisited.
    2001.
"""

class Benchmark:
    def __init__(self):
        self.files = [file_name.replace(".txt","") for file_name
            in os.listdir(DATA_ROOT) if ".txt" in file_name]
        self.dataset=defaultdict(list)
        for file_name in self.files:
            for line in open(DATA_ROOT + "/" + file_name + ".txt"):
                self.dataset[file_name].append([ float(w) if i == 2 else w
                    for i, w in enumerate(line.strip().split())])

    @staticmethod
    def cos(vec1,vec2):
        return vec1.dot(vec2)/(linalg.norm(vec1)*linalg.norm(vec2))

    @staticmethod
    def rho(vec1,vec2):
        return stats.stats.spearmanr(vec1, vec2)[0]

    @staticmethod
    def pprint(result, title):
        from prettytable import PrettyTable
        table = PrettyTable(["Dataset", "Found", "Not Found", "Score"])
        table.title = 'Results for {}'.format(title)
        table.align["Dataset"] = "l"
        for k, v in result.items():
            table.add_row([k,v[0],v[1],v[2]])
        print(table)

    @staticmethod
    def pprint_compare(results, methods, title):
        assert len(results) == len(methods)
        from prettytable import PrettyTable
        table = PrettyTable(["Score", "EN-RG-65", "EN-WS-353-ALL",
            "MSR-analogy"])
        table.title = 'Results for {} dataset'.format(title)
        for result, method in zip(results, methods):
            table.add_row([method, list(result.values())[1][2],
                list(result.values())[0][2], list(result.values())[2][2]])
        print(table)

    def evaluate(self, E, title, discount_query_words=False, batch_size=200,
        print=True):
        """
        Evaluates RG-65, WS-353 and MSR benchmarks


        :param object E: WordEmbedding object.
        :param string title: Title of the results table.
        :param int batch_size: Size of the batches in which to process
            the queries.
        :param boolean discount_query_words: Give analogy solutions that appear
            in the query 0 score in MSR benchmark. (Default = False)
        :param boolean print: Print table with results. (Default = True)
        :returns: dict with results
        """
        word_dict = E.get_dict()
        result = {}
        vocab = word_dict.keys()
        for file_name, data in self.dataset.items():
            pred, label, found, notfound = [] ,[], 0, 0
            for datum in data:
                if datum[0] in vocab and datum[1] in vocab:
                    found += 1
                    pred.append(self.cos(word_dict[datum[0]],
                        word_dict[datum[1]]))
                    label.append(datum[2])
                else:
                    notfound += 1
            result[file_name] = [found, notfound, self.rho(label,pred)*100]
        msr_res = self.MSR(E, discount_query_words, batch_size)
        result["MSR-analogy"] = [msr_res[1], msr_res[2], msr_res[0]]
        if print:
            self.pprint(result, title)
        return result

    def MSR(self, E, discount_query_words=False, batch_size=200):
        """
        Executes MSR-analogy benchmark on the word embeddings in E


        :param object E: WordEmbedding object.
        :param boolean discount_query_words: Give analogy solutions that appear
            in the query 0 score. (Default = False)
        :param int batch_size: Size of the batches in which to process
            the queries.
        :returns: Percentage of correct analogies (accuracy),
            number of queries without OOV words,
            number of queries with OOV words
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

        # Batch the queries up
        y = []
        n_batches = (len(analogy_answers) // batch_size)+1
        for batch in np.array_split(filtered_questions, n_batches):
            # Extract relevant embeddings from E
            a = E.vecs[np.vectorize(E.index.__getitem__)(batch[:,0])]
            x = E.vecs[np.vectorize(E.index.__getitem__)(batch[:,1])]
            b = E.vecs[np.vectorize(E.index.__getitem__)(batch[:,2])]
            all_y = E.vecs

            # Calculate scores
            batch_pos = ((1+all_y@x.T)/2)*((1+all_y@b.T)/2)
            batch_neg = (1+all_y@a.T+0.00000001)/2
            batch_scores = batch_pos/batch_neg

            # If set, set scores of query words to 0
            if discount_query_words:
                query_ind = np.vectorize(E.index.__getitem__)(batch).T
                batch_scores[query_ind, np.arange(
                    batch_scores.shape[1])[None,:]] = 0


            # Retrieve words with best analogy scores
            y.append(np.array(E.words)[np.argmax(batch_scores, axis=0)])

        # Calculate returnable metrics
        y = np.hstack(y)[:,None]
        accuracy = np.mean(y==filtered_answers)*100
        words_not_found = len(analogy_answers) - len(filtered_answers)

        return accuracy, len(filtered_answers), words_not_found

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_filename",
        help="The name of the embedding")
    parser.add_argument("--table_title", type=str, default="benchmark",
        help="Title of the printed table")

    print_parser = parser.add_mutually_exclusive_group(required=False)
    print_parser.add_argument('--print', dest='print_t', action='store_true')
    print_parser.add_argument('--dont-print', dest='print_t',
        action='store_false')
    parser.set_defaults(print_t=True)

    query_parser = parser.add_mutually_exclusive_group(required=False)
    query_parser.add_argument('--discard-query-words', dest='dqw',
        action='store_true')
    query_parser.add_argument('--dont-discard', dest='dqw',
        action='store_false')
    parser.set_defaults(dqw=False)

    args = parser.parse_args()
    print(args)

    E = we.WordEmbedding(args.embedding_filename)
    B = Benchmark()

    results = B.evaluate(E, args.table_title, args.dqw, args.print_t)
    if not args.print_t:
        print(results)
