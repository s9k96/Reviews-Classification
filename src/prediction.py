from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
import os
import pickle

model = keras.models.load_model('models/model-lstm/model-lstm.h5')
with open('models/model-lstm/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f) 
# print(model.summary())


def predict_probability(s):

    tokens = tokenizer.texts_to_sequences([s])
    tokens = sequence.pad_sequences(tokens, maxlen=1000)
    return model.predict(tokens)


def predict_sentiment(s):
    """Predicting Sentiments given a Review.

    Args:
        s (str): Review text

    Returns:
        sentiment: positive/negative
    """
    pred = predict_probability(s)[0][0]
    print(s, pred)
    return "negative" if pred < 0.5 else "positive"