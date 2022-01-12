import re, numpy as np, pandas as pd
from pprint import pprint

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])


def show_coherence_and_perplexity(model, corpus):
    coh_model = CoherenceModel(model=model, corpus=corpus, coherence='u_mass')
    coher_val = coh_model.get_coherence()  # get coherence value

    print("Coherence value: " + str(coher_val))

    perpl_values = []
    counter = 1
#    for text in corpus:
#        perpl_values.append(model.log_perplexity(text))
#        print(str(counter) + ". Perplexity value: " + str(perpl_values.pop()))
#        counter += 1



def show_table_with_documents_stats(dataFrame, text_numbers = None):

    # Create temporary copy of data frame to not change original one
    if text_numbers is None:
        text_numbers = []
    temp_df = dataFrame.copy(deep=True)
    if ("Text" in temp_df):
        temp_df = temp_df.drop(columns="Text")

    # Limit number of docs to not overflow the window
    if not bool(text_numbers) or len(text_numbers) > 20:
        temp_df = temp_df.head(10)
    else:
        temp_df = temp_df.iloc[text_numbers]

    colLabels = temp_df.columns

    # Choose first 6 words (or less, if they aren't there) of the text
    #if ('Text' in temp_df):
    #    temp_df['Text'] = temp_df['Text'].apply(lambda x: x.rsplit(maxsplit=len(x.split()) - 6)[0])
    #elif ('Representative Text' in temp_df):
    #    temp_df['Representative Text'] = temp_df['Representative Text'].apply(
    #        lambda x: x.rsplit(maxsplit=len(x.split()) - 6)[0])
    #else:
    #    print("Wrong data frame")
    #    return

    plt.rcParams["figure.figsize"] = [16, 3]
    plt.rcParams["figure.autolayout"] = True
    fig, axs = plt.subplots(1, 1)
    axs.axis('off')

    the_table = axs.table(cellText=temp_df.values,
                          colLabels=colLabels,
                          loc='upper center',
                          cellLoc='center')

    the_table.auto_set_column_width(col=list(range(len(colLabels))))
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)

    plt.title('Documents statistics', fontdict=dict(size=22))
    plt.show()


def show_statistics(lda_model, corpus, texts, text_numbers):
#    texts = [text[:60] + "..." for text in texts_in]
    def format_topics_sentences(ldamodel=None, corpus=corpus, texts=texts):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row_list in enumerate(ldamodel[corpus]):
            row = row_list[0] if ldamodel.per_word_topics else row_list
            # print(row)
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

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return (sent_topics_df)

    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=texts)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', "Text"]

#    df_dominant_topic.head(10)

    text_numbers = list(set(text_numbers))


    show_table_with_documents_stats(df_dominant_topic, text_numbers)

    show_coherence_and_perplexity(model=lda_model, corpus=corpus)


    # Display setting to show more characters in column
    pd.options.display.max_colwidth = 100

    sent_topics_sorteddf_mallet = pd.DataFrame()
    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
    sent_topics_outdf_grpd = sent_topics_outdf_grpd

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                                 grp.sort_values(['Perc_Contribution'], ascending=False).head(1)],
                                                axis=0)

    # Reset Index
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

    # Format
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]
    sent_topics_sorteddf_mallet = sent_topics_sorteddf_mallet.drop(columns="Text")

    # Show
#    sent_topics_sorteddf_mallet.head(10)



    show_table_with_documents_stats(sent_topics_sorteddf_mallet)





    # Words can be count as amount of " "(spacebar) + 1
    doc_lens = [d.count(' ')+1 for d in texts]

    if (len(corpus) > 1):
        values, counts = np.unique(doc_lens, return_counts=True)

        # Plot
        plt.figure(figsize=(16, 7), dpi=160)
    #    plt.hist(doc_lens, bins=(max(doc_lens) - min(doc_lens) + 1), color='navy')

        plt.axis([0, round(max(doc_lens) * 1.05, 0), 0, round(max(counts) * 1.5, 0) + 1])

    #    plt.hist(doc_lens, bins=round(len(values) * 2) + 1, color='navy')
        plt.hist(doc_lens, bins=round(len(doc_lens)) + 1, color='navy')

        plt.text(0.85 * max(doc_lens), 0.9 * plt.ylim()[1], "Mean   : " + str(round(np.mean(doc_lens))))
        plt.text(0.85 * max(doc_lens), 0.8 * plt.ylim()[1], "Median : " + str(round(np.median(doc_lens))))
        plt.text(0.85 * max(doc_lens), 0.7 * plt.ylim()[1], "Stdev  : " + str(round(np.std(doc_lens))))
        plt.text(0.85 * max(doc_lens), 0.6 * plt.ylim()[1], "1%ile  : " + str(round(np.quantile(doc_lens, q=0.01))))
        plt.text(0.85 * max(doc_lens), 0.5 * plt.ylim()[1], "99%ile : " + str(round(np.quantile(doc_lens, q=0.99))))

        plt.ylabel("Number of Documents")
        plt.xlabel("Document Word Count")
    #    plt.axis([0, round(max(doc_lens)*1.05, 0), 0, len(corpus)])

        plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
        plt.show()








    cols = [color for name, color in mcolors.XKCD_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS' / mcolors.TABLEAU_COLORS

    rows = int(np.sqrt(lda_model.num_topics))
    columns = int(lda_model.num_topics/rows)+1


    fig, axes = plt.subplots(rows, columns, figsize=(16, 14), dpi=160, sharex="none", sharey="none")

    max_len_ticks = int(round(max(doc_lens) * 1.05, 0))
    max_ticks_range = max(3, 10 - lda_model.num_topics)

    for i, ax in enumerate(axes.flatten()):

        # delete unused subplots
        if i >= lda_model.num_topics:
            plt.delaxes(axes[int(i / columns)][i % columns])
            continue

        df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]
        doc_lens = [d.count(' ')+1 for d in df_dominant_topic_sub.Text]

        if bool(doc_lens):
            max_len = int(round(max(doc_lens) * 1.05, 0))
            min_len = min(doc_lens)
            values, counts = np.unique(doc_lens, return_counts=True)
        else:
            max_len = 1
            min_len = 0
            values, counts = [[0], [0]]


#        ax.hist(doc_lens, bins=(max_len - min_len + 1), color=cols[i])
#        ax.hist(doc_lens, bins=len(doc_lens)*10+1, color=cols[i])
        ax.hist(doc_lens, bins=(len(values) * 2 + 1), color=cols[i])
        ax.tick_params(axis='y', labelcolor="black", color=cols[i])
        ax.set(xlim=(min_len, max_len), ylim=(0, max(1, max(counts))), xlabel='Document Word Count')
        ax.set_ylabel('Number of Documents', color="black")
        ax.set_title('Topic: ' + str(i), fontdict=dict(size=16, color=cols[i]))
        sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())


    fig.tight_layout()
    plt.subplots_adjust()
    fig.subplots_adjust(top=0.90)
    plt.xticks(np.linspace(0, max_len_ticks, max_ticks_range))
    fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=22)
    plt.show()



def sentences_chart(lda_model, corpus, start = 0, end = 1):
    if end == 1:
        end = len(corpus)
    if start > end:
        start = end - 10
    if start < 0:
        start = 0


    corp = corpus[start:end]
    mycolors = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    fig, axes = plt.subplots(end-start, 1, figsize=(20, (end-start)*0.95), dpi=160)
    if (end-start) == 1:
        axes.axis('off')
    else:
        axes[0].axis('off')
    for i, ax in enumerate(axes):
        corp_cur = corp[i]
        topic_percs, wordid_topics, wordid_phivalues = lda_model[corp_cur]
        word_dominanttopic = [(lda_model.id2word[wd], topic[0]) for wd, topic in wordid_topics]
        ax.text(0.01, 0.5, "Doc " + str(i) + ": ", verticalalignment='center',
                fontsize=16, color='black', transform=ax.transAxes, fontweight=700)

        # Draw Rectange
        topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True)
        ax.add_patch(Rectangle((0.0, 0.05), 0.99, 0.90, fill=None, alpha=1,
                               color=mycolors[topic_percs_sorted[0][0]], linewidth=2))

        word_pos = 0.085
        for j, (word, topics) in enumerate(word_dominanttopic):
            if j < 8:
                ax.text(word_pos, 0.5, word,
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=16, color=mycolors[topics],
                        transform=ax.transAxes, fontweight=700)
                word_pos += .015 * len(word)  # to move the word for the next iter
                ax.axis('off')
        ax.text(word_pos, 0.5, '. . .',
                horizontalalignment='left',
                verticalalignment='center',
                fontsize=16, color='black',
                transform=ax.transAxes)


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle('Sentence Topic Coloring for Documents: ' + str(start) + ' to ' + str(end-1), fontsize=22, y=0.95, fontweight=700)
    plt.tight_layout()
    plt.show()





def t_SNE_clustering(lda_model, corpus):
    topic_weights = []
    for i, row_list in enumerate(lda_model[corpus]):
        topic_weights.append([w for i, w in row_list[0]])

    # Array of topic weights
    arr = pd.DataFrame(topic_weights).fillna(0).values

    # Keep the well separated points (optional)
    arr = arr[np.amax(arr, axis=1) > 0.35]

    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)

    # Plot the Topic Clusters using Bokeh
    output_file("t_SNE_clusters.html")
    n_topics = lda_model.num_topics
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics),
                  plot_width=900, plot_height=700)
    plot.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=mycolors[topic_num])
    show(plot)
