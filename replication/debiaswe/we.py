from __future__ import print_function, division
import re
import sys
import os
import numpy as np
import scipy.sparse
from sklearn.decomposition import PCA
from debiaswe.download import download
from debiaswe.embeddings_config import ID
if sys.version_info[0] < 3:
    import io
    open = io.open
else:
    unicode = str
"""
Tools for debiasing word embeddings.
Extended from the code from:

Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""


def dedup(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def safe_word(w):
    # ignore words with numbers, etc.
    # [a-zA-Z\.'_\- :;\(\)\]] for emoticons
    return (re.match(r"^[a-z_]*$", w) and len(w) < 20
        and not re.match(r"^_*$", w))


def to_utf8(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`),
        to bytestring in utf8."""
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')


class WordEmbedding:
    def __init__(self, embedding, limit=None):
        self.thresh = None
        self.max_words = None
        self.desc = embedding

        from_file = False
        fname = None

        # If embedding in standard available embeddings, check if download
        # is needed.
        if embedding in ID.keys():
            extension = ID[embedding]["extension"]
            fname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                "..", "embeddings", embedding + extension)
            # If embedding file not present, download it
            if not os.path.exists(fname):
                download(embedding)
        else:
            # Check for file path
            assert os.path.exists(embedding) and os.path.isfile(embedding), \
                "Not an available embedding or known file path"
            # If valid file available, load from that file
            from_file = True
            fname = embedding
            print(f"Creating embedding from file: {os.path.abspath(embedding)}")

        # Load binary files using gensim
        if fname.endswith(".bin"):
            import gensim.models
            model = gensim.models.KeyedVectors.load_word2vec_format(fname,
                binary=True, limit=limit)
            self.words = sorted([w for w in model.vocab],
                key=lambda w: model.vocab[w].index)
            self.vecs = np.array([model[w] for w in words], dtype='float32')
        # Load non binary files by reading line by line
        else:
            vecs = []
            words = []
            # Open and read from file
            with open(fname, "r", encoding='utf8') as f:
                for line in f:
                    if len(words) == limit:
                        break
                    s = line.split()
                    v = np.array([float(x) for x in s[1:]])
                    words.append(s[0])
                    vecs.append(v)
            # Determine correct (i.e. most common) vector length
            lengths = [len(v) for v in vecs]
            correct_length = max(lengths, key=lengths.count)
            # Filter out any loaded vectors with uncommon length
            vecs_filtered = []
            for v in vecs:
                if len(v) == correct_length:
                    vecs_filtered.append(v)
                elif from_file:
                    print("Got weird line:", line)
            self.vecs = np.array(vecs_filtered, dtype='float32')
            self.words = words

        # If needed, reindex and normalize after loading
        self.reindex()
        norms = np.linalg.norm(self.vecs, axis=1)
        if max(norms)-min(norms) > 0.0001:
            if from_file: print("Normalizing vectors...")
            self.normalize()
        print("Embedding shape:", self.vecs.shape)

    def get_dict(self):
        return {key:value for key, value in zip(self.words, self.vecs)}

    def reindex(self):
        self.index = {w: i for i, w in enumerate(self.words)}
        self.n, self.d = self.vecs.shape
        assert self.n == len(self.words) == len(self.index)
        self._neighbors = None
        print(self.n, "words of dimension", self.d, ":", ", ".join(
            self.words[:4] + ["..."] + self.words[-4:]))

    def v(self, word):
        return self.vecs[self.index[word]]

    def diff(self, word1, word2):
        v = self.vecs[self.index[word1]] - self.vecs[self.index[word2]]
        return v/np.linalg.norm(v)

    def normalize(self):
        self.desc += ", normalize"
        self.vecs /= np.linalg.norm(self.vecs, axis=1)[:, np.newaxis]
        self.reindex()

    def shrink(self, numwords):
        self.desc += ", shrink " + str(numwords)
        self.filter_words(lambda w: self.index[w]<numwords)

    def filter_words(self, test):
        """
        Keep some words based on test, e.g. lambda x: x.lower()==x
        """
        self.desc += ", filter"
        kept_indices, words = zip(*[[i, w] for i, w
            in enumerate(self.words) if test(w)])
        self.words = list(words)
        self.vecs = self.vecs[kept_indices, :]
        self.reindex()

    def save(self, filename):
        with open(filename, "w", encoding="utf8") as f:
            f.write("\n".join([w+" " + " ".join([str(x) for x in v]) for w, v
                in zip(self.words, self.vecs)]))
        print("Wrote", self.n, "words to", filename)

    def save_w2v(self, filename, binary=True):
        with open(filename, 'wb', encoding="utf8") as fout:
            fout.write(to_utf8("%s %s\n" % self.vecs.shape))
            # store in sorted order: most frequent words at the top
            for i, word in enumerate(self.words):
                row = self.vecs[i]
                if binary:
                    fout.write(to_utf8(word) + b" " + row.tostring())
                else:
                    fout.write(to_utf8("%s %s\n" % (word, ' '.join("%f" % val
                        for val in row))))

    def remove_directions(self, directions): #directions better be orthogonal
        self.desc += ", removed"
        for direction in directions:
            self.desc += " "
            if type(direction) is np.ndarray:
                v = direction / np.linalg.norm(direction)
                self.desc += "vector "
            else:
                w1, w2 = direction
                v = self.diff(w1, w2)
                self.desc += w1 + "-" + w2
            self.vecs = self.vecs - self.vecs.dot(v)[:, None].dot(v[None, :])
        self.normalize()

    def compute_neighbors_if_necessary(self, thresh, max_words):
        thresh = float(thresh) # dang python 2.7!
        if (self._neighbors is not None and self.thresh == thresh
            and self.max_words == max_words):
            return
        print("Computing neighbors")
        self.thresh = thresh
        self.max_words = max_words
        vecs = self.vecs[:max_words]
        dots = vecs.dot(vecs.T)
        dots = scipy.sparse.csr_matrix(dots * (dots >= 1-thresh/2))
        from collections import Counter
        rows, cols = dots.nonzero()
        nums = list(Counter(rows).values())
        print("Mean:", np.mean(nums)-1)
        print("Median:", np.median(nums)-1)
        rows, cols, vecs = zip(*[(i, j, vecs[i]-vecs[j]) for i, j, x
            in zip(rows, cols, dots.data) if i<j])
        self._neighbors = rows, cols, np.array([v/np.linalg.norm(v)
            for v in vecs])
        print(self._neighbors[2].shape)

    def neighbors(self, word, thresh=1):
        dots = self.vecs.dot(self.v(word))
        return [self.words[i] for i, dot in enumerate(dots)
            if dot >= 1-thresh/2]

    def more_words_like_these(self, words, topn=50, max_freq=100000):
        v = sum(self.v(w) for w in words)
        dots = self.vecs[:max_freq].dot(v)
        thresh = sorted(dots)[-topn]
        words = [w for w, dot in zip(self.words, dots) if dot>=thresh]
        return sorted(words, key=lambda w: self.v(w).dot(v))[-topn:][::-1]

    def best_analogies_dist_thresh(self, v, thresh=1, topn=500,
        max_words=50000):
        """Metric is cos(a-c, b-d) if |b-d|^2 < thresh, otherwise 0
        """
        vecs, vocab = self.vecs[:max_words], self.words[:max_words]
        self.compute_neighbors_if_necessary(thresh, max_words)
        rows, cols, vecs = self._neighbors
        scores = vecs.dot(v/np.linalg.norm(v))
        pi = np.argsort(-abs(scores))

        ans = []
        usedL = set()
        usedR = set()
        for i in pi:
            if abs(scores[i])<0.001:
                break
            row = rows[i] if scores[i] > 0 else cols[i]
            col = cols[i] if scores[i] > 0 else rows[i]
            if row in usedL or col in usedR:
                continue
            usedL.add(row)
            usedR.add(col)
            ans.append((vocab[row], vocab[col], abs(scores[i])))
            if len(ans)==topn:
                break

        return ans

    def profession_stereotypes(self, profession_words, bias_space, print_firstn=20):
        # Calculate the projection values onto the bias subspace
        sp = sorted([(self.v(w).dot(bias_space), w) for w in profession_words if w in self.words])
        # Check what genders belong to positive/negative projection values
        pos_neg = ("Female", "Male") if self.v("she").dot(bias_space) > 0 else ("Male", "Female")
        # Print the professions with scores
        print(pos_neg[0].center(38) + "|" + pos_neg[1].center(38))
        print("-"*77)
        for i in range(min(print_firstn, len(sp))):
            print(str(sp[-(i+1)][0].round(3)).ljust(8) # score negative
                + sp[-(i+1)][1].rjust(29) + " | "       # profession negative
                + sp[i][1].ljust(29)                    # score positive
                + str(sp[i][0].round(3)).rjust(8))     # profession positive
        return sp


def viz(analogies):
    print("\n".join(str(i).rjust(4)+a[0].rjust(29) + " | " + a[1].ljust(29) +
        (str(a[2]))[:4] for i, a in enumerate(analogies)))


def text_plot_words(xs, ys, words, width = 90, height = 40, filename=None):
    PADDING = 10 # num chars on left and right in case words spill over
    res = [[' ' for i in range(width)] for j in range(height)]
    def rescale(nums):
        a = min(nums)
        b = max(nums)
        return [(x-a)/(b-a) for x in nums]
    print("x:", (min(xs), max(xs)), "y:",(min(ys),max(ys)))
    xs = rescale(xs)
    ys = rescale(ys)
    for (x, y, word) in zip(xs, ys, words):
        i = int(x*(width - 1 - PADDING))
        j = int(y*(height-1))
        row = res[j]
        z = list(row[i2] != ' ' for i2 in range(max(i-1, 0),
            min(width, i + len(word) + 1)))
        if any(z):
            continue
        for k in range(len(word)):
            if i+k>=width:
                break
            row[i+k] = word[k]
    string = "\n".join("".join(r) for r in res)
#     return string
    if filename:
        with open(filename, "w", encoding="utf8") as f:
            f.write(string)
        print("Wrote to", filename)
    else:
        print(string)


def doPCA(pairs, embedding, num_components = 10):
    matrix = []
    for a, b in pairs:
        center = (embedding.v(a) + embedding.v(b))/2
        matrix.append(embedding.v(a) - center)
        matrix.append(embedding.v(b) - center)
    matrix = np.array(matrix)
    pca = PCA(n_components = num_components)
    pca.fit(matrix)
    # bar(range(num_components), pca.explained_variance_ratio_)
    return pca


def drop(u, v):
    return u - v * (u.dot(v) / v.dot(v))
