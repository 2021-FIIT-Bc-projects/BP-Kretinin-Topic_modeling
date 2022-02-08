# Copyright 2022 Mykyta Kretinin
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import logging
import re, spacy
import joblib

from gensim.test.utils import datapath, get_tmpfile
import gensim.corpora
from gensim.corpora import Dictionary, WikiCorpus, MmCorpus
from gensim.utils import simple_preprocess

gensim.corpora.wikicorpus.ARTICLE_MIN_WORDS = 50
gensim.corpora.wikicorpus.IGNORED_NAMESPACES = ['Wikipedia', 'Category', 'File', 'Portal', 'Template', 'MediaWiki', 'User', 'Help', 'Book', 'Draft', 'WikiProject', 'Special', 'Talk']
gensim.corpora.wikicorpus.RE_P0 = re.compile('<!--.*?-->', re.DOTALL)
gensim.corpora.wikicorpus.RE_P1 = re.compile('<ref([> ].*?)(</ref>|/>)', re.DOTALL)
gensim.corpora.wikicorpus.RE_P10 = re.compile('<math([> ].*?)(</math>|/>)', re.DOTALL)
gensim.corpora.wikicorpus.RE_P11 = re.compile('<(.*?)>', re.DOTALL)
gensim.corpora.wikicorpus.RE_P12 = re.compile('(({\\|)|(\\|-(?!\\d))|(\\|}))(.*?)(?=\\n)')
gensim.corpora.wikicorpus.RE_P13 = re.compile('(?<=(\\n[ ])|(\\n\\n)|([ ]{2})|(.\\n)|(.\\t))(\\||\\!)([^[\\]\\n]*?\\|)*')
gensim.corpora.wikicorpus.RE_P14 = re.compile('\\[\\[Category:[^][]*\\]\\]')
gensim.corpora.wikicorpus.RE_P15 = re.compile('\\[\\[([fF]ile:|[iI]mage)[^]]*(\\]\\])')
gensim.corpora.wikicorpus.RE_P16 = re.compile('\\[{2}(.*?)\\]{2}')
gensim.corpora.wikicorpus.RE_P17 = re.compile('(\\n.{0,4}((bgcolor)|(\\d{0,1}[ ]?colspan)|(rowspan)|(style=)|(class=)|(align=)|(scope=))(.*))|(^.{0,2}((bgcolor)|(\\d{0,1}[ ]?colspan)|(rowspan)|(style=)|(class=)|(align=))(.*))')
gensim.corpora.wikicorpus.RE_P2 = re.compile('(\\n\\[\\[[a-z][a-z][\\w-]*:[^:\\]]+\\]\\])+$')
gensim.corpora.wikicorpus.RE_P3 = re.compile('{{([^}{]*)}}', re.DOTALL)
gensim.corpora.wikicorpus.RE_P4 = re.compile('{{([^}]*)}}', re.DOTALL)
gensim.corpora.wikicorpus.RE_P5 = re.compile('\\[(\\w+):\\/\\/(.*?)(( (.*?))|())\\]')
gensim.corpora.wikicorpus.RE_P6 = re.compile('\\[([^][]*)\\|([^][]*)\\]', re.DOTALL)
gensim.corpora.wikicorpus.RE_P7 = re.compile('\\n\\[\\[[iI]mage(.*?)(\\|.*?)*\\|(.*?)\\]\\]')
gensim.corpora.wikicorpus.RE_P8 = re.compile('\\n\\[\\[[fF]ile(.*?)(\\|.*?)*\\|(.*?)\\]\\]')
gensim.corpora.wikicorpus.RE_P9 = re.compile('<nowiki([> ].*?)(</nowiki>|/>)', re.DOTALL)

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

# function to save processed wiki dump as (corpora, texts) object
def ready_corpus(texts, corpora):
    transformed_corpora = tuple([doc] for doc in corpora)
    transformed_texts = []
    for doc in texts:
        text = ""
        for word in doc:
            text += str(word) + " "
        transformed_texts.append(text[:-1])
    transformed_texts = tuple(doc for doc in transformed_texts)

    zip_obj = zip(transformed_corpora, transformed_texts)

    zip_list = list(zip_obj)

    filename = '../../data/corpora/corpus_' + str(len(transformed_corpora)) + '_files.jl'
    joblib.dump(zip_list, filename)
    print("Corpora saved at " + filename)


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # 1.6 MB zip
#    path_to_wiki_dump = datapath("enwiki-latest-pages-articles1.xml-p000000010p000030302-shortened.bz2")

    # 28 MB zip
#    path_to_wiki_dump = datapath("enwiki-latest-pages-articles15.xml-p17324603p17460152.bz2")

    # 350 MB zip
#    path_to_wiki_dump = datapath("enwiki-latest-pages-articles-multistream13.xml-p9172789p10672788.bz2")

    # 107 MB zip
#    path_to_wiki_dump = datapath("enwiki-latest-pages-articles-multistream24.xml-p56564554p57025655.bz2")

    # 507 MB zip
    path_to_wiki_dump = datapath("enwiki-latest-pages-articles-multistream11.xml-p5399367p6899366.bz2")



    wiki = WikiCorpus(path_to_wiki_dump, processes=7, lower=True, token_min_len=3, token_max_len=15)

    texts = []
    for text in wiki.get_texts():
        texts.append(text)

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
    data_lemmatized = lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    corpora = joblib.load("../../data/processed/proc_wiki_data_500MB.jl")

    # "proc_wiki_data.jl" - for very small (1.6 MB) dump, for functionality testing
    # "proc_wiki_data2.jl" - smaller (28 MB) dump
    # "proc_wiki_data3.jl" - bigger (350 MB) dump
    # "proc_wiki_data_100MB.jl" - medium (107 MB) dump
    # "proc_wiki_data_500MB.jl" - big (507 MB) dump
    joblib.dump(data_lemmatized, '../../data/processed/proc_wiki_data.jl')

    # uncomment, if processed wiki dump shall be saved as (corpora, texts) object in /data/corpora/ dir
#    ready_corpus(texts, corpora)
    print("done")

if __name__ == '__main__':
    main()