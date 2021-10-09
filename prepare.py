import sys

import os
from bs4 import BeautifulSoup
import requests
from zipfile import ZipFile
from tempfile import TemporaryFile
import numpy as np
import pandas as pd
import re, nltk, spacy, gensim
from spacy.lang.en import English
import pyLDAvis.gensim_models

import joblib

from gensim.utils import simple_preprocess

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

import matplotlib.pyplot as plt


REQUIRED_PYTHON = "python3"

# Define function to predict topic for a given text document.
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


def get_top_topic(ldamodel, corpus):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Format
    df_dominant_topic = sent_topics_df.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords']
#    print(df_dominant_topic.head(10))





def generate_HTML_doc_LDA(ldamodel, corpus):
    visualisation = pyLDAvis.gensim_models.prepare(ldamodel, corpus, ldamodel.id2word)
    pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')





def get_corpus(lda_model, text):
    global sent_to_words
    global lemmatization

    # Step 1: Clean with simple_preprocess
    mytext_2 = list(sent_to_words(text))

    # Step 2: Lemmatize
    mytext_3 = lemmatization(mytext_2, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Step 3: LDA update
    corpus = [lda_model.id2word.doc2bow(text) for text in mytext_3]
    lda_model.update(corpus)

    return corpus[0]


def sent_to_words(sentences):
    # Remove Emails
    sentences = [re.sub('\S*@\S*\s?', '', sent) for sent in sentences]

    # Remove most of special characters
    sentences = [re.sub(r"[^a-zA-Z0-9]+", " ", sent) for sent in sentences]

    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if
                                   token.pos_ in allowed_postags]))
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]
    return texts_out


#Shows documents' topic probabilities distribution (upper) and keywords (lower) tables
def show_LDA_results_as_table(cell_text, keywords):
    colLabels = ['Topic ' + str(i) for i in range(len(cell_text[0]))]
    colLabels2 = ['Word ' + str(i) for i in range(len(keywords[0]))]
    rowLabels = ['test_ ' + str(i) for i in range(len(cell_text))]

    if len(colLabels) >=len(colLabels2):
        top_length = len(colLabels)
    else:
        top_length = len(colLabels2)

    if(len(rowLabels)<3):
        plt.rcParams["figure.figsize"] = [top_length, 3]
    else:
        plt.rcParams["figure.figsize"] = [top_length, 3 + len(rowLabels) * 0.33]
    plt.rcParams["figure.autolayout"] = True
    fig, axs = plt.subplots(1, 1)
    axs.axis('off')

    the_table = axs.table(cellText=cell_text,
                          rowLabels=rowLabels,
                          colLabels=colLabels,
                          loc='upper center')
    the_table2 = axs.table(cellText=keywords,
                          rowLabels=rowLabels,
                          colLabels=colLabels2,
                          loc='lower center')

    the_table.auto_set_column_width(col=list(range(len(colLabels))))
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table2.auto_set_column_width(col=list(range(len(colLabels))))
    the_table2.auto_set_font_size(False)
    the_table2.set_fontsize(10)
    plt.autoscale()
    plt.show()


#Shows keywords of the topics of LDA model
def show_lda_model_topics(df_topic_keywords):

    cell_text = []
    for row in range(len(df_topic_keywords.index)):
        cell_text.append(df_topic_keywords.iloc[row])

    df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]

    plt.rcParams["figure.figsize"] = [len(df_topic_keywords.columns), 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig, axs = plt.subplots(1, 1)
    axs.axis('off')
    the_table = axs.table(cellText=cell_text,
                          rowLabels=df_topic_keywords.index,
                          colLabels=df_topic_keywords.columns,
                          loc='center')

    the_table.auto_set_column_width(col=list(range(len(df_topic_keywords.columns))))
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    plt.autoscale()
    plt.show()


#  https://towardsdatascience.com/web-scraping-news-articles-in-python-9dd605799558
def get_last_N_articles_from_bbc(number):
    url = "https://www.bbc.com/news/world"
    r1 = requests.get(url)
    coverpage = r1.content

    soup1 = BeautifulSoup(coverpage, "html5lib")

    coverpage_news = soup1.find_all(
        class_='gs-c-promo-heading gs-o-faux-block-link__overlay-link gel-pica-bold nw-o-link-split__anchor')

    # Scraping the first N articles
    number_of_articles = number

    # Main page ref
    main_page = "https://www.bbc.com"

    # Empty lists for content, links and titles
    news_contents = []
    list_links = []
    list_titles = []

    zipArch = ZipFile('data/external/texts/sample.zip', 'w')
    tmp_file_path = "data/external/texts/"

    for iter in np.arange(0, number_of_articles):

        # only news articles (there are also albums and other things)
        #        if "inenglish" not in coverpage_news[n].find('a')['href']:
        #            continue

        # Getting the link of the article
        link = main_page + coverpage_news[iter]['href']
        list_links.append(link)

        # Getting the title
        title = coverpage_news[iter].get_text()
        list_titles.append(title)

        # Reading the content (it is divided in paragraphs)
        article = requests.get(link)
        article_content = article.content
        soup_article = BeautifulSoup(article_content, 'html5lib')
        body = soup_article.find_all('article', class_='ssrcss-1mc1y2-ArticleWrapper e1nh2i2l6')

        # Skip if page doesn't have "article" part
        if len(body) == 0:
            continue
        x = body[0].find_all('p')

        # Unifying the paragraphs
        list_paragraphs = []
        final_article = ""

        # Store text of article, if article contents it
        if len(x) > 0:
            for p in np.arange(0, len(x)):
                paragraph = x[p].get_text()
                list_paragraphs.append(paragraph)
                final_article = " ".join(list_paragraphs)

            news_contents.append(final_article)

            # Open file(create), write text inside it, move its copy to the zip, and delete this file
            tmp_file = open(tmp_file_path + "Article" + str(iter) + ".txt", "w", encoding="utf-8")
            tmp_file.write(final_article)
            tmp_file.close()
            zipArch.write(tmp_file.name, os.path.basename(tmp_file.name))
            os.remove(tmp_file_path + "Article" + str(iter) + ".txt")

    # close the Zip File
    zipArch.close()

