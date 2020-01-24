''' This file is used to download different word embeddings.

DISCLAIMER: This code is largely copied from this stack overflow answer:
https://stackoverflow.com/a/39225272

The following word embeddings are available:
- word2vec_large
- word2vec_small
- glove_small
- fasttext_large
- fasttext_small

TODO: Embedding descriptions (e.g. embedding dimensions, training data)

'''

import requests
import os
from debiaswe.embeddings_config import ID

def download(embedding):
    assert embedding in ID.keys(), "Unkown embedding."

    URL = "https://docs.google.com/uc?export=download"
    id = ID[embedding]["id"]
    extension = ID[embedding]["extension"]

    # Destination is current file destination, one directory up, then the
    # "embeddings" directory.
    destination = os.path.dirname(os.path.abspath(__file__)) \
        + "\\..\\embeddings\\" + embedding + extension
    print(f"Downloading {embedding} embedding to {os.path.abspath(destination)}")

    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)