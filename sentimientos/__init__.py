# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 15:04:00 2018

@author: alessandro gianfelici
"""

from tensorflow.keras.layers import Bidirectional, Dense, Embedding, Input, LSTM, Dropout, Flatten, AveragePooling1D, Conv1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import spacy
import os
import string
import nltk
from nltk.corpus import stopwords

path = os.path.dirname(__file__)

MAX_SEQUENCE_LENGTH = 35
EMBEDDING_DIM = 300
MAX_N_WEMBS = 200000
nlp = spacy.load('es_core_news_sm')
NB_WEMBS = MAX_N_WEMBS

nltk.download('stopwords')

def getStopWords(custom_stop_words : list = []) -> list:
    """
    This function returns a list of spanish stop words
    """
    return stopwords.words('spanish') + list(string.printable) + custom_stop_words
    
def lemmatize(text : str, nlp = nlp):
    """
    Input:
        - text : str = spanish text to lemmatize
        - nlp (optional) = modello spacy per l'italiano
    Output:
        - testo lemmatizzato (str)
    """
    doc = nlp(text)
    #use list comprehension to reduce time and resources needed
    return " ".join([token.lemma_ for token in doc]).lower()

def process_texts(texts, maxlen):
    texts_feats = map(lambda x: create_features(x, maxlen), texts)
    return texts_feats   
    
def create_features(text, maxlen, myVoc):
    wemb_idxs = []
    for myWord in text:
        wemb_idx = myVoc.get(myWord, myVoc['unknown'])
        wemb_idxs.append(wemb_idx)
    wemb_idxs = pad_sequences([wemb_idxs], maxlen=maxlen, value=NB_WEMBS)
    return [wemb_idxs]

def removePunct(stringa):
    """
    Rimuove la punteggiatura da una stringa
    """
    
    punteggiatura = [symb for symb in string.punctuation]
    for punct in punteggiatura:
        stringa = str(stringa).replace(punct, " ")
    return stringa

def removeStoWords(stringa):
    """
    Rimuove le stopwords
    """
    for stop in getStopWords():
        stringa = str(stringa).replace(stop, " ")
    return stringa


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

#================ Model ======================#

class Model():
    def __init__(self, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
        self.load_model(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
        self.early_stop =  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)
        self.epochs =  10^4

    def buildModel(self, dict_dim=NB_WEMBS, max_len=MAX_SEQUENCE_LENGTH, emb_dim=EMBEDDING_DIM):

        x1 = Input(shape=(max_len,))
        w_embed = Embedding(dict_dim+1, emb_dim, input_length=max_len, trainable=False)(x1)
        w_embed = Dropout(0.5)(Dense(64, activation='relu')(w_embed))
        h = Conv1D(filters=32, kernel_size=2, padding='valid', activation='relu')(w_embed)
        h = Bidirectional(LSTM(32, return_sequences=True, recurrent_dropout=0.5), merge_mode='concat')(h)
        h = AveragePooling1D(pool_size=2, strides=None, padding='valid')(h)
        h = Bidirectional(LSTM(16, return_sequences=True, recurrent_dropout=0.5), merge_mode='concat')(h)
        h = AveragePooling1D(pool_size=2, strides=None, padding='valid')(h)
        h = Flatten()(h)
        preds_pol = Dense(2, activation='sigmoid')(h)

        model_pol = Model(inputs=[x1], outputs=preds_pol)
        model_pol.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
        #model_pol.summary()
        self.model = model_pol
        return model_pol

    def load_weights(self):
        pass

    def fit(self, x, y, **args) -> tf.keras.callbacks.History:
        """
        Fit the deep learning model
        """
        if self.model is None:
            self.model = self.buildModel(x.shape[1])
        return self.model.fit(
                x, 
                y, 
                epochs=self.epochs,
                validation_split = 0.1,  
                callbacks=[self.early_stop], 
                **args)
    
    def predict(self, x : list) -> list:
        """
        Predict
        """
        return self.model.predict(x)

def calculate_polarity(sentences: list):
    results = []
    sentences = list(map(lambda x: x.lower(), sentences))
    #sentences = list(map(lambda x: re.sub('[^a-zA-z0-9\s]','',x), sentences))
    X_ctest = list(process_texts(sentences, MAX_SEQUENCE_LENGTH))
    n_ctest_sents = len(X_ctest)

    test_wemb_idxs = np.reshape(np.array([e[0] for e in X_ctest]), [n_ctest_sents, MAX_SEQUENCE_LENGTH])

    sent_model = Model(dict_dim=NB_WEMBS, max_len=MAX_SEQUENCE_LENGTH, emb_dim=EMBEDDING_DIM)
    preds = sent_model.predict([test_wemb_idxs])
    for i in range(n_ctest_sents):
        results.append(sentences[i] + ' - ' + 'opos: ' + str(preds[i][0]) + ' - oneg: ' + str(preds[i][1]))
    return results, preds