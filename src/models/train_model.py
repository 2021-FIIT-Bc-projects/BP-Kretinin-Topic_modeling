import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import gensim.corpora as corpora
import gensim
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt



def main():
    print("Choose source of texts for learning:\n 1) Newsgrounds\n 2) Wikipedia\n")
    in_val = input()
    if in_val == '1':

        data_lemmatized = joblib.load('../../data/processed/proc_data2.jl')

    elif in_val == '2':

        data_lemmatized = joblib.load('../../data/processed/proc_wiki_data2.jl')

    else:
        print("Invalid argument")
        return -1

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=4,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=10,
                                               passes=10,
                                               alpha='symmetric',
                                               iterations=100,
                                               per_word_topics=True)

    pprint(lda_model.print_topics())

    print(lda_model.id2word)  # Model attributes

    # Visualize the topics
    visualisation = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')

    if in_val == '1':
        joblib.dump(lda_model, '../../models/lda_model2.jl')
        # then reload it with
        lda_model = joblib.load('../../models/lda_model2.jl')
    elif in_val == '2':
        joblib.dump(lda_model, '../../models/lda_wiki_model.jl')
        # then reload it with
        lda_model = joblib.load('../../models/lda_wiki_model.jl')

if __name__ == '__main__':
    main()