# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import pandas as pd
import re, nltk, spacy, gensim
from gensim.utils import simple_preprocess
from spacy.lang.en import English

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

import joblib



def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Import Dataset

    #first big training dataset
#    df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')

    #first dataset(20) for visualization from newsgrounds
    df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
    df = df.loc[df.target_names.isin(['soc.religion.christian', 'rec.sport.hockey', 'talk.politics.mideast', 'rec.motorcycles']), :]
    print(df.shape)  # > (2361, 3)
    df.head()

    joblib.dump(df, '../../data/raw/raw_data_vis.jl')
    # then reload it with
    df = joblib.load('../../data/raw/raw_data_vis.jl')






    # Convert to list
    data = df.content.values.tolist()

    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]

    # print(data[:1])

    def sent_to_words(sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(data))

    # print(data_words[:1])

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if
                                       token.pos_ in allowed_postags]))
        texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]
        return texts_out

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # Run in terminal: python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Do lemmatization keeping only Noun, Adj, Verb, Adverb
    data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    joblib.dump(data_lemmatized, '../../data/processed/proc_data2.jl')
    # then reload it with
    data_lemmatized = joblib.load('../../data/processed/proc_data2.jl')

    print("Done")






if __name__ == '__main__':
    main()
