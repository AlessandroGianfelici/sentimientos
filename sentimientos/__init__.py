# -*- coding: utf-8 -*-
"""
Created on Fry Sep 17 08:04:00 2021

@author: alessandro gianfelici
"""

from tensorflow.keras.layers import Bidirectional, Dense, Embedding, Input, LSTM, Dropout, Flatten, AveragePooling1D, Conv1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import spacy
import os, shutil
import string
import nltk
from nltk.corpus import stopwords
from toolz.functoolz import pipe
import yaml
import logging
import re
import pandas as pd
from zipfile import ZipFile, ZIP_DEFLATED
import requests

logger = logging.getLogger('__main__')
path = os.path.dirname(__file__)

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300
MAX_N_WEMBS = 500000
nlp = spacy.load('es_core_news_sm')
NB_WEMBS = MAX_N_WEMBS

MODEL_ID = "1_9Y1Nes_m4TQEuBkfPn-1FVVDVofbhBm"
MODEL_PATH = os.path.join(path, 'saved_model.zip')
FOLDER_PATH = MODEL_PATH[:-4]
nltk.download('stopwords')


def loadVocabulary():
    logger.info("Loading vocabulary...This will take a wile")
    with open(os.path.join(path, f'vocabulary.yaml'), 'r', encoding="utf-8") as handler:
        return yaml.load(handler, Loader=yaml.FullLoader)

try:
    VOCABULARY = loadVocabulary()
except:
    print("vocabulary.yaml not found, check the path or train again the model!")
    VOCABULARY = None

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

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

def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def getStopWords(custom_stop_words : list = []) -> list:
    """
    This function returns a list of spanish stop words
    """
    return  + custom_stop_words
    
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

def process_texts(texts, myVoc : dict, maxlen=MAX_SEQUENCE_LENGTH):
    return map(lambda x: create_features(x, myVoc, maxlen= maxlen), texts)
    
def create_features(text, myVoc, maxlen):
    wemb_idxs = []
    for myWord in text:
        wemb_idx = myVoc.get(myWord, NB_WEMBS)
        wemb_idxs.append(wemb_idx)
    wemb_idxs = pad_sequences([wemb_idxs], maxlen=maxlen, value=NB_WEMBS)
    return [wemb_idxs]

def removePunct(stringa: str) -> str:
    """
    Rimuove la punteggiatura da una stringa
    """
    
    punteggiatura = [symb for symb in string.punctuation]
    for punct in punteggiatura:
        stringa = str(stringa).replace(punct, " ")
    return stringa


def tokenize(myString : str, nlp=nlp) -> str:
    doc = nlp(myString)
    return [str(token).lstrip().rstrip() for token in doc]

def getStopWords():
    return stopwords.words('spanish') + list(string.printable) + ["  "]

def removeStopWords(tokenizedStr : list) -> list:
    """
    Rimuove le stopwords
    """
    return list(filter(lambda x : x not in getStopWords(), tokenizedStr))

def removeNumeric(tokenizedStr : list) -> list:
    """
    Rimuove le stopwords
    """
    return list(filter(lambda x : not(hasNumbers(x)), tokenizedStr))


def preprocessTexts(myString : str) -> list:
    return pipe(myString, lambda x : x.lower().replace("\n", " ").replace("\r", " ").replace("\t", " "),
                          removePunct,
                          lemmatize,
                          tokenize,
                          removeNumeric,
                          removeStopWords)


def file_folder_exists(path: str):
    """
    Return True if a file or folder exists.

    :param path: the full path to be checked
    :type path: str
    """
    try:
        os.stat(path)
        return True
    except:
        return False

def download_and_extract(file_id, destination):
    download_file_from_google_drive(file_id, destination)
    shutil.unpack_archive(destination, destination[:-4])
    return destination[:-4]

class SentimientosModel():
    def __init__(self, max_len=MAX_SEQUENCE_LENGTH, emb_dim=EMBEDDING_DIM, pretrained=True):
        FOLDER_PATH = MODEL_PATH[:-4]
        if pretrained:
            if not file_folder_exists(FOLDER_PATH):
                print("I am loading the pretrained model, this will take a while!")
                FOLDER_PATH = download_and_extract(MODEL_ID, MODEL_PATH)
            self.model = tf.keras.models.load_model(FOLDER_PATH)
        else:
            self.model = None
        #self.load_model(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
        self.early_stop =  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.epochs =  10000
        self.dict_dim=NB_WEMBS
        self.max_len = max_len
        self.emb_dim=emb_dim

    def buildModel(self):

        x1 = Input(shape=(self.max_len,))
        
        preds_pol = pipe(x1, 
        Embedding(self.dict_dim+1, self.emb_dim, input_length=self.max_len, trainable=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Conv1D(filters=32, kernel_size=2, padding='valid', activation='relu'),
        Bidirectional(LSTM(32, return_sequences=True, recurrent_dropout=0.5), merge_mode='concat'),
        AveragePooling1D(pool_size=2, strides=None, padding='valid'),
        Bidirectional(LSTM(16, return_sequences=True, recurrent_dropout=0.5), merge_mode='concat'),
        AveragePooling1D(pool_size=2, strides=None, padding='valid'),
        Flatten(),
        Dense(2, activation='sigmoid'))
        
        model_pol = Model(inputs=[x1], outputs=preds_pol)
        model_pol.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[tf.keras.metrics.AUC()])
        #model_pol.summary()
        self.model = model_pol
        return model_pol

    def fit(self, x, y, **args) -> tf.keras.callbacks.History:
        """
        Fit the deep learning model
        """
        if self.model is None:
            self.model = self.buildModel()
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

    def save(self, filename : str):
        self.model.save(filename)
        return zipFolder(filename)

    def plotHistory(self, history: tf.keras.callbacks.History, title='', full_filename = None) -> None:
        """
        Plot training vs validation loss over epoch.
        """
        import matplotlib.pyplot as plt

        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        plt.figure()
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(f'{self.loss} over epoch')
        plt.plot(hist['epoch'], hist[self.loss],
               label='Train Error')
        plt.plot(hist['epoch'], hist[f'val_{self.loss}'],
               label = 'Val Error')
        plt.legend()
        if full_filename is not None:
            plt.savefig(full_filename, dpi=300)
        plt.show(block=False)

def get_all_file_paths(directory): 
  
    # initializing empty file paths list 
    file_paths = [] 
  
    # crawling through directory and subdirectories 
    for root, _, files in os.walk(directory): 
        for filename in files: 
            # join the two strings in order to form the full filepath. 
            filepath = os.path.join(root, filename) 
            file_paths.append(filepath) 
  
    # returning all file paths
    return file_paths

def remove_prefix(text : str, prefix : str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever

def zipFolder(directory : str) -> str: 

    # calling function to get all file paths in the directory 
    file_paths = get_all_file_paths(directory) 
  
    # printing the list of all files to be zipped 
    logger.info(f'{directory} folder will be zipped:') 
 
    _, rete = os.path.split(directory)
    pathtree = os.path.commonpath(file_paths)

    with ZipFile(f'{directory}.zip','w') as zip: 
        # writing each file one by one 
        for file in file_paths:
            arcname = os.path.join(rete, remove_prefix(file, pathtree))
            logger.info(f"Writing file {file} as {arcname}")
            zip.write(file, arcname=arcname) 
  
    logger.info('All files zipped successfully!')
    return f'{directory}.zip'


def calculate_polarity(sentences: list):
    results = []
    sentences = list(map(lambda x: x.lower(), sentences))
    X_ctest = list(process_texts(sentences, VOCABULARY, MAX_SEQUENCE_LENGTH))
    n_ctest_sents = len(X_ctest)
    test_wemb_idxs = np.reshape(np.array([e[0] for e in X_ctest]), [n_ctest_sents, MAX_SEQUENCE_LENGTH])
    sent_model = SentimientosModel(max_len=MAX_SEQUENCE_LENGTH, emb_dim=EMBEDDING_DIM, pretrained=True)
    preds = sent_model.predict([test_wemb_idxs])
    for i in range(n_ctest_sents):
        results.append(sentences[i] + ' - ' + 'opos: ' + str(preds[i][0]) + ' - oneg: ' + str(preds[i][1]))
    return results, preds
