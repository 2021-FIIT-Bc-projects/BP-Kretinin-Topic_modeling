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



def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Import Dataset
#    data_lemmatized = joblib.load('../../data/processed/proc_data2.jl')
#    wiki_corpus = joblib.load('../../data/processed/proc_wiki_data2.jl')


    # 1.6 MB zip
#    path_to_wiki_dump = datapath("enwiki-latest-pages-articles1.xml-p000000010p000030302-shortened.bz2")

    # 28 MB zip
    path_to_wiki_dump = datapath("enwiki-latest-pages-articles15.xml-p17324603p17460152.bz2")

#    corpus_path = get_tmpfile("wiki-corpus.mm")

    wiki = WikiCorpus(path_to_wiki_dump, processes=4, lower=True, token_min_len=3, token_max_len=15)  # create word->word_id mapping, ~8h on full wiki
#    MmCorpus.serialize(corpus_path, wiki)  # another 8h, creates a file in MatrixMarket format and mapping

    # Save WikiCorpus object (include dictionary with token2id array)
#    joblib.dump(wiki, '../../data/processed/proc_wiki_data.jl')

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


    # "proc_wiki_data.jl" - for smaller (1.6 MB) dump
    # "proc_wiki_data2.jl" - for bigger (28 MB) dump
    joblib.dump(data_lemmatized, '../../data/processed/proc_wiki_data2.jl')

    print("done")

if __name__ == '__main__':
    main()