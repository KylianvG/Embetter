from __future__ import print_function, division
import debiaswe.we as we
import json
import numpy as np
import argparse
import sys
import torch
if sys.version_info[0] < 3:
    import io
    open = io.open
"""
Hard-debias and soft-debias for word embeddings.
Extended from the code from:

Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""


def hard_debias(E, gender_specific_words, definitional, equalize):
    """
    Hard debiases word embeddings.


    :param object E: WordEmbedding object.
    :param list gender_specific_words: List of gender specific words, which are
        not dibiased.
    :param list definitional: List containing lists of corresponding
        definitional words.
    :param list equalize: List containing lists of corresponding words that
        should only differ in gender.
    :returns: None
    """
    gender_direction = we.doPCA(definitional, E).components_[0]
    specific_set = set(gender_specific_words)
    for i, w in enumerate(E.words):
        if w not in specific_set:
            E.vecs[i] = we.drop(E.vecs[i], gender_direction)
    E.normalize()
    candidates = {x for e1, e2 in equalize for x in [(e1.lower(), e2.lower()),
                                                     (e1.title(), e2.title()),
                                                     (e1.upper(), e2.upper())]}
    print(candidates)
    for (a, b) in candidates:
        if (a in E.index and b in E.index):
            y = we.drop((E.v(a) + E.v(b)) / 2, gender_direction)
            z = np.sqrt(1 - np.linalg.norm(y)**2)
            if (E.v(a) - E.v(b)).dot(gender_direction) < 0:
                z = -z
            E.vecs[E.index[a]] = z * gender_direction + y
            E.vecs[E.index[b]] = -z * gender_direction + y
    E.normalize()

def soft_debias(E, gender_specific_words, definitional, log=True):
    """
    Soft debiases word embeddings.


    :param object E: WordEmbedding object.
    :param list gender_specific_words: List of gender specific words, which are
        not dibiased.
    :param list definitional: List containing lists of corresponding
        definitional words.
    :param bool log: Print optimizer progress.
    :returns: None
    """
    W = torch.from_numpy(E.vecs).t()
    neutrals = list(set(E.words) - set(gender_specific_words))
    neutrals = torch.tensor([E.vecs[E.index[w]] for w in neutrals]).t()
    gender_direction = torch.tensor([we.doPCA(definitional, E).components_[0]]).t()
    l = 0.2 # lambda
    u, s, _ = torch.svd(W)
    s = torch.diag(s)

    # precompute
    t1 = s.mm(u.t())
    t2 = u.mm(s)

    transform = torch.randn(300, 300, requires_grad=True)
    epochs = 2000
    optimizer = torch.optim.Adam([transform], lr=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,1500,1800], gamma=0.1)
    best = (None, float("inf")) # (best transform, lowest loss)

    for i in range(epochs):
        optimizer.zero_grad()
        TtT = torch.mm(transform.t(), transform)
        norm1 = (t1.mm(TtT - torch.eye(300)).mm(t2)).norm(p="fro")
        norm2 = (neutrals.t().mm(TtT).mm(gender_direction)).norm(p="fro")
        loss = norm1 + l * norm2
        if loss.item() < best[1]:
            best = (transform, loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if i % 10 == 0:
            if log:
                print("Loss @ Epoch #" + str(i) + ":", loss.item())
            
    transform = best[0].detach()
    if log:
        print(f"Lowest loss: {best[1]}")

    debiased_embeds = transform.mm(W).t().numpy()
    debiased_embeds = debiased_embeds / np.linalg.norm(debiased_embeds, axis=1)[:, None]
    E.vecs = debiased_embeds


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("embedding_filename", help="The name of the embedding")
    parser.add_argument("definitional_filename", help="JSON of definitional pairs")
    parser.add_argument("gendered_words_filename", help="File containing words not to neutralize (one per line)")
    parser.add_argument("equalize_filename", help="???.bin")
    parser.add_argument("debiased_filename", help="???.bin")

    args = parser.parse_args()
    print(args)

    with open(args.definitional_filename, "r") as f:
        defs = json.load(f)
    print("definitional", defs)

    with open(args.equalize_filename, "r") as f:
        equalize_pairs = json.load(f)

    with open(args.gendered_words_filename, "r") as f:
        gender_specific_words = json.load(f)
    print("gender specific", len(gender_specific_words), gender_specific_words[:10])

    E = we.WordEmbedding(args.embedding_filename)

    print("Debiasing...")
    debias(E, gender_specific_words, defs, equalize_pairs)

    print("Saving to file...")
    if args.embedding_filename[-4:] == args.debiased_filename[-4:] == ".bin":
        E.save_w2v(args.debiased_filename)
    else:
        E.save(args.debiased_filename)

    print("\n\nDone!\n")
