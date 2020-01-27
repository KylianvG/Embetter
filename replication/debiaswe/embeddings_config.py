''' Configuration file for downloading available pretrained word embeddings.

ID is a dictionary with word embeddings as keys, and dictionaries containing a
Google Drive download id and file extension as values.

Google Drive folder:
https://drive.google.com/drive/folders/1AY6IwIoJqepxw3s6wK-Fl6udMkVmMjFO

Note that the embedding names in de Google Drive folder do not exactly match the
embedding names used in the ID dictionary.
'''

ID = {
    ########## Word2Vec ##########
    # 300d embeddings trained on Google News, 3M words
    "word2vec_large" : {
        "id" : "1ep-6TjdfG86EbdcypvrHg7oxnc_3aD02",
        "extension" : ".bin"
    },
    # 300d embeddings trained on Google News, 26423 words
    "word2vec_small" : {
        "id" : "1Lk-jzberOG9F9lA0pnEyzlF9mGEyk8OZ",
        "extension" : ".txt"
    },
    # 300d embeddings trained on Google News, 26423 words, hard debiased
    "word2vec_small_hard_debiased" : {
        "id" : "0",
        "extension" : ".txt"
    },
    # 300d embeddings trained on Google News, 26423 words, soft debiased
    "word2vec_small_soft_debiased" : {
        "id" : "0",
        "extension" : ".txt"
    },

    ########## GloVe ##########
    # 300d embeddings, 42982 words
    "glove_small" : {
        "id" : "1cZ5UG5LmjCM5vNczLHFnDb7zP7tgce0r",
        "extension" : ".txt"
    },
    # 300d embeddings, 42982 words, hard debiased
    "glove_small_hard_debiased" : {
        "id" : "0",
        "extension" : ".txt"
    },
    # 300d embeddings, 42982 words, soft debiased
    "glove_small_soft_debiased" : {
        "id" : "0",
        "extension" : ".txt"
    },

    ########## fastText ##########
    # 768d embeddings
    "fasttext_large" : {
        "id" : "1G23SP2D7qKISGNKH5HK6v9Su3wo-jOxH",
        "extension" : ".vec"
    },
    # 768d embeddings, 27014 words
    "fasttext_small" : {
        "id" : "1rLnn9MtIHwvCcRo9JUbaDvEZb-maygX_",
        "extension" : ".txt"
    },
    # 768d embeddings, 27014 words, hard debiased
    "fasttext_small_hard_debiased" : {
        "id" : "0",
        "extension" : ".txt"
    },
    # 768d embeddings, 27014 words, soft debiased
    "fasttext_small_soft_debiased" : {
        "id" : "0",
        "extension" : ".txt"
    }
}
