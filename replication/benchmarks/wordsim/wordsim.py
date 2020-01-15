import os 
import sys
import logging
import argparse
import numpy
from collections import defaultdict
from scipy import linalg, mat, dot, stats
DATA_ROOT = os.path.dirname( os.path.abspath( __file__ ) ) + "/data/"

class Wordsim:
    def __init__(self,lang):
        logging.info("collecting datasets ..")
        self.files = [ file_name.replace(".txt","") for file_name in os.listdir(DATA_ROOT+lang) if ".txt" in file_name ]
        self.dataset=defaultdict(list)
        for file_name in self.files:
            for line in open(DATA_ROOT + lang + "/" + file_name + ".txt"):
                self.dataset[file_name].append([ float(w) if i == 2 else w for i, w in enumerate(line.strip().split())])

    @staticmethod
    def cos(vec1,vec2):
        return vec1.dot(vec2)/(linalg.norm(vec1)*linalg.norm(vec2))

    @staticmethod
    def rho(vec1,vec2):
        return stats.stats.spearmanr(vec1, vec2)[0]

    @staticmethod
    def load_vector(path, binary=False):
        print("load_vector")
        word2vec = {}
        if binary:
            import gensim.models
            model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
            words = sorted([w for w in model.vocab], key=lambda w: model.vocab[w].index)
            vecs = [model[w] for w in words]
            for i, word in enumerate(words):
                word2vec[word] = vecs[i]
        else:
            try:
                print("loading vector...")
                logging.info("loading vector ..")
                if path[-3:] == ".gz":
                    import gzip
                    f = gzip.open(path, "rb")
                else:
                    f = open(path, "rb")
            except ValueError:
                print("Oops!  No such file.  Try again ..")
        
            for wn, line in enumerate(f): 
                line = line.lower().strip()
                word = line.split()[0]
                print(word)
                
                word2vec[word] = numpy.array(list(map(float,line.split()[1:])))
                # else:
                # word2vec[word] = numpy.array(map(float,line.split()[1:]))
        print("loaded vector {0} words found ..".format(len(word2vec.keys())))
        logging.info("loaded vector {0} words found ..".format(len(word2vec.keys())))
        return word2vec

    @staticmethod
    def pprint(result):
        from prettytable import PrettyTable
        x = PrettyTable(["Dataset", "Found", "Not Found", "Score (rho)"])
        x.align["Dataset"] = "l"
        for k, v in result.items():
            x.add_row([k,v[0],v[1],v[2]])
        print(x)

    @staticmethod
    def pprint_w2vnews(result1, result2):
        from prettytable import PrettyTable
        x = PrettyTable(["", "EN-RG-65", "EN-WS-353-ALL"])
        x.align["Dataset"] = "l"
        x.add_column("", ["Before", "Hard-debiased"])
        x.add_column("", [result1['EN-RG-65'][2], result2['EN-RG-65'][2]])
        x.add_column("", [result1['EN-WS-353-ALL'][2], result2['EN-WS-353-ALL'][2]])
        print(x)

    def evaluate(self, word_dict):
        result = {}
        # word_dict = {key.decode('utf-8'): value for (key, value) in word_dict.items()}
        vocab = word_dict.keys()
        print(vocab)
        for file_name, data in self.dataset.items():
            pred, label, found, notfound = [] ,[], 0, 0
            for datum in data:
                if datum[0] in vocab and datum[1] in vocab:
                    found += 1
                    pred.append(self.cos(word_dict[datum[0]],word_dict[datum[1]]))
                    label.append(datum[2])
                else:
                    notfound += 1
            result[file_name] = (found, notfound, self.rho(label,pred)*100)
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', '-l', default="en")
    parser.add_argument('--vector', '-v', default="")
    args = parser.parse_args()
    wordsim = Wordsim(args.lang)
    word2vec = wordsim.load_vector(args.vector)
    result = wordsim.evaluate(word2vec)
    wordsim.pprint(result)
