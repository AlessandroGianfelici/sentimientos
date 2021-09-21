from sentimientos import loadVocabulary, process_texts, preprocessTexts, SentimientosModel, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, MAX_N_WEMBS, NB_WEMBS
import pandas as pd
import os
import yaml
import numpy as np

path = os.path.dirname(__file__)

def getTrainData() -> pd.DataFrame:
    url = r'https://drive.google.com/u/0/uc?id=1Vy2pu3wx-7EkNlvDLIUSUeu5NWnuSOGu&export=download'
    review_data = pd.read_csv(os.path.join(path, "train_data", "reviews", "raw_data.txt"))[['review_text', 'review_stars']]
    wikipedia_data = pd.read_csv(url, compression='zip')
    review_data = review_data.rename(columns={'review_text' : 'text'})
    review_data = review_data.loc[review_data['review_stars'] != 3]
    review_data.loc[review_data['review_stars'] < 3, 'sentiment'] = 'negative'
    review_data.loc[review_data['review_stars'] > 3, 'sentiment'] = 'positive'
    full_dataset = pd.concat([review_data[["text", "sentiment"]], wikipedia_data]).set_index('text')
    return pd.get_dummies(full_dataset['sentiment'])[['positive', 'negative']].reset_index()


def buildVocabulary(tokenized_text):
    import gensim
    myvoc = gensim.corpora.dictionary.Dictionary(documents=tokenized_text)
    dictionary = myvoc.token2id
    frequencyDict = myvoc.cfs
    newDict = dict(filter(lambda elem: elem[1] > 2, frequencyDict.items()))
    mydict = dict(filter(lambda elem: elem[1] in newDict.keys(), dictionary.items()))
    shrinkedDict = {key : value for key, value in zip(mydict.keys(), range(len(mydict)))}
    shrinkedDict['unknown'] = len(shrinkedDict)
    return shrinkedDict


if __name__ == '__main__':
    try:
        train_data = pd.read_pickle(os.path.join(path, 'sentimientos', "serialized_train_data.pickle"))
        mydict = loadVocabulary()
    except:
        train_data = getTrainData()
        print(f"Collected {len(train_data)} texts! {sum(train_data['positive'])} positive and {sum(train_data['negative'])} negative")
        train_data['processed_text'] = train_data['text'].map(preprocessTexts)
        mydict = buildVocabulary(train_data['processed_text'])
        train_data.to_pickle(os.path.join(path, 'sentimientos', "serialized_train_data.pickle"))
        with open(os.path.join(path, 'sentimientos', 'vocabulary.yaml'), 'w') as outfile:
            yaml.dump(mydict, outfile, default_flow_style=False)
    print("done!")
    train_data = train_data.sample(frac = 1)
    X_train = list(process_texts(train_data['processed_text'], mydict, maxlen=MAX_SEQUENCE_LENGTH))
    Y_train =  train_data[['positive','negative']].values
    n_ctest_sents = len(X_train)
    X_train_reshaped = np.reshape(np.array([e[0] for e in X_train]), [n_ctest_sents, MAX_SEQUENCE_LENGTH])
    sent_model = SentimientosModel(max_len=MAX_SEQUENCE_LENGTH, emb_dim=EMBEDDING_DIM, pretrained=False)
    sent_model.fit(X_train_reshaped, Y_train, shuffle=True, batch_size=10000)
    sent_model.save(os.path.join(path, 'sentimientos', "saved_model"))