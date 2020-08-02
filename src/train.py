import torch
import tensorflow as tf
import pandas as pd
import config
import dataset
import io
import psutil 

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

    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.review.values.tolist())

    xtrain = tokenizer.texts_to_sequences(train_df.review.values)
    xtest = tokenizer.texts_to_sequences(valid_df.review.values)


    xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=config.MAX_LEN)
    xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest, maxlen=config.MAX_LEN)

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

    del df, xtrain, xtest
    embedding_dict = load_vectors('../models/crawl-300d-2M.vec')

if __name__ == '__main__':
    print(psutil.virtual_memory())
    df = pd.read_csv('../data/imdb-folds.csv')
    run(df, fold=1)    
