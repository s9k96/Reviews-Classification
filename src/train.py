import torch
# import tensorflow as tf
import pandas as pd
import config
import dataset
import io, os
import psutil 
from tokenizer import Tokenizer
from pprint import pprint
import numpy as np

def create_embedding_matrix(word_index, embedding_dict):
    """
    This function creates the embedding matrix.   
    :param word_index: a dictionary with word:index_value   
    :param embedding_dict: a dictionary with word:embedding_vector   
    :return: a numpy array with embedding vectors for all known words 
    """   
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():         
        if word in embedding_dict:          
               embedding_matrix[i] = embedding_dict[word]
    return embedding_matrix 
 
 

def load_vectors(fname):
    # taken from: https://fasttext.cc/docs/en/english-vectors.html
    fin = io.open(
        fname,
        'r',
        encoding='utf-8',
        newline='\n',
        errors='ignore'
    )
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def run(df, fold):

    train_df = df[df.kfold != fold].reset_index(drop = True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    print(len(train_df))
    print(len(valid_df))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df.review.values.tolist(), info=True)

    xtrain = tokenizer.texts_to_sequences(train_df.review.values)
    xtest = tokenizer.texts_to_sequences(valid_df.review.values)


    xtrain = tokenizer.pad_sequences(xtrain, max_len = 200)
    xtest = tokenizer.pad_sequences(xtest, max_len = 200)
    print(xtrain.shape)
    print(xtest.shape)

    train_dataset = dataset.Dataset(
        reviews = xtrain, 
        targets = train_df.sentiment.values
    )

    valid_dataset = dataset.Dataset(
        reviews = xtest, 
        targets = valid_df.sentiment.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size = config.BATCH_SIZE,
        num_workers = 2
    )    

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size = config.BATCH_SIZE,
        num_workers = 2
    )    
    print(psutil.virtual_memory())

    if not os.path.exists("../data/embedding_matrix.npy"):
        embedding_dict = load_vectors('../models/crawl-300d-2M.vec')
        print("Loaded Vectors")
        embedding_matrix = create_embedding_matrix(tokenizer.word_index, embedding_dict)
        print("Created Matrix")

        np.save("../data/embedding_matrix.npy", embedding_matrix)


if __name__ == '__main__':
    print(psutil.virtual_memory())
    df = pd.read_csv('../data/imdb-folds.csv')
    run(df, fold=1)    
