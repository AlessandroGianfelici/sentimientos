from sentimientos import preprocessTexts, Model, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, MAX_N_WEMBS
import pandas as pd
import os
import gensim
import yaml

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
    myvoc = gensim.corpora.dictionary.Dictionary(documents=tokenized_text)
    dictionary = myvoc.token2id
    frequencyDict = myvoc.cfs
    newDict = dict(filter(lambda elem: elem[1] > 2, frequencyDict.items()))
    result = dict(filter(lambda elem: elem[1] in newDict.keys(), dictionary.items()))
    result['unknown'] = len(result)
    return result


if __name__ == '__main__':
    try:
        train_data = pd.read_pickle("serialized_train_data.pickle")
        mydict = buildVocabulary(train_data['processed_text'])
    except:
        train_data = getTrainData()
        print(f"Collected {len(train_data)} texts! {sum(train_data['positive'])} positive and {sum(train_data['negative'])} negative")
        train_data['processed_text'] = train_data['text'].map(preprocessTexts)
        mydict = buildVocabulary(train_data['processed_text'])
        train_data.to_pickle("serialized_train_data.pickle")
        with open('vocabulary.yml', 'w') as outfile:
            yaml.dump(mydict, outfile, default_flow_style=False)
    