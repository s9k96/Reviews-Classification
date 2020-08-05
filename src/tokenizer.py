import re
import numpy as np

class Tokenizer:
    '''
    Tokenizer: Convert text to integer tokens

    '''
    def __init__(self):
        # self.num_words = num_words
        self.word_index = dict()
        
    def fit_on_texts(self, texts, info=False):
        '''
        texts: list of strings
        '''
        unique_words = set()
        for i in texts:
            unique_words.update(i.split())
        if info:
            print("Length of input: ", len(texts))
        
        for i, word in enumerate(unique_words):
            word = re.sub('[^A-Za-z0-9]+', '', str(word).lower())
            if self.word_index.get(word, -1) == -1:
                self.word_index[word] = i
        if info:
            print("Unique words: ", len(self.word_index))
        
    def texts_to_sequences(self, texts):
        '''
        texts: list of strings
        '''
        tokens = []
        for text in texts:
            # text = text[0]
            sentence = []
            for word in text.split():
                word = re.sub('[^A-Za-z0-9]+', '', str(word).lower())
                # print(word, self.word_index.get(word, -1))
                sentence.append(self.word_index.get(word, -1))
            # print(sentence)
            tokens.append(sentence)
        return tokens

    
    
    def pad_sequences(self, tokens, max_len):
        '''
        tokens: np.array. output from texts_to_sequence
        '''
        # max_len = max([len(i) for i in tokens])
        data = np.zeros(shape = (len(tokens), max_len))
        for i, line in enumerate(tokens):
            line = np.array(line)[:max_len]
            length_of_line  = len(line)
            # print(length_of_line, max_len-length_of_line)
            a = np.array([0]*(max_len-length_of_line))
            data[i, :] = np.concatenate([a, line])
        return data
            

