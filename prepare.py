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
from datetime import date

import joblib

from gensim.utils import simple_preprocess

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['reuter', 'oct', 'from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

import matplotlib.pyplot as plt


REQUIRED_PYTHON = "python3"

# Define function to predict topic for a given text document.
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# The same one as in the test_environment.py
def progress_bar(iteration, total):
    total_len = 100
    percent_part = ("{0:.2f}").format(100 * (iteration / total))
    filled = int(total_len * iteration / total)
    bar = '█' * filled + '-' * (total_len - filled)
    print(f'\r Progress: [{bar}] {percent_part}%', end='')
    if iteration == total:
        print()


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
#    corpus = [lda_model.id2word.doc2bow(text) for text in mytext_3]

    # Calculate and print perplexity
#    print("Perplexity: " + str(lda_model.log_perplexity(corpus)))

#    lda_model.update(corpus)

    return mytext_3


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
def get_last_N_articles_from_reuters(number):
    print("If progress bar don't move for a long time, then 'class' parameter for the BeatuifulSoup find_all() should be revised and actualized")
#    current_page = "https://www.reuters.com/news/archive/worldNews?view=page&page=100&pageSize=10"
    current_page = "https://www.reuters.com/news/archive/worldNews?view=page&page=1&pageSize=10"

    processed_num = 0

    page_counter = 1

    # Main page ref
    main_page = "https://www.reuters.com"
    page_first_part = "https://www.reuters.com/news/archive/worldNews?view=page&page="
    page_second_part = "&pageSize=10"

    # Empty lists for content, links and titles
    news_contents = []
    list_links = []
    list_titles = []

    zipArch = ZipFile('data/external/texts/articles_' + str(date.today()) + "_" + str(number) + '.zip', 'w')
    tmp_file_path = "data/external/texts/"

    while (processed_num < number):

        r1 = requests.get(current_page)
        coverpage = r1.content

        soup1 = BeautifulSoup(coverpage, "html5lib")

        coverpage_news = soup1.find_all(class_='story-content')

        # Getting the link of the article
        #        current_page = main_page + page_next[0]['href']

        page_counter += 1

        current_page = page_first_part + str(page_counter) + page_second_part

        # each page contains 10 articles
        for iter in np.arange(0, 10):
            progress_bar(processed_num, number)
            if (processed_num == number):
                break

            # only news articles (there are also albums and other things)
            #        if "inenglish" not in coverpage_news[n].find('a')['href']:
            #            continue

            # Getting the link of the article
            link = main_page + coverpage_news[iter].contents[1]['href']
            list_links.append(link)

            # Getting the title
            title = coverpage_news[iter].get_text()
            list_titles.append(title)

            # Reading the content (it is divided in paragraphs)
            article = requests.get(link)
            article_content = article.content
            soup_article = BeautifulSoup(article_content, 'html5lib')
            regex = re.compile('article-body__content___.*')
#            body = soup_article.find_all('div', class_='ArticleBody__content___2gQno2 paywall-article')
            body = soup_article.find_all('div', {"class": regex})

            # Skip if page doesn't have "article" part
            if len(body) == 0:
                continue
            x = body[0].find_all('p')

            # Unifying the paragraphs
            list_paragraphs = []
            final_article = ""

            # Store text of article, if article contents it
            if len(x) > 0:
                processed_num += 1
                for p in np.arange(0, len(x)):
                    paragraph = x[p].get_text()
                    list_paragraphs.append(paragraph)
                final_article = " ".join(list_paragraphs)

                news_contents.append(final_article)

                # Open file(create), write text inside it, move its copy to the zip, and delete this file
                tmp_file = open(tmp_file_path + "Article" + str(processed_num) + ".txt", "w", encoding="utf-8")
                tmp_file.write(final_article)
                tmp_file.close()
                zipArch.write(tmp_file.name, os.path.basename(tmp_file.name))
                os.remove(tmp_file_path + "Article" + str(processed_num) + ".txt")

    # close the Zip File
    zipArch.close()

