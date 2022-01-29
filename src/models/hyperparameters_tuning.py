import joblib

import gensim.corpora as corpora
import gensim

import time

from test_environment import progress_bar

from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt


# coherence_scores - array of dictionary items with with "num_topics" and "coherence" keys
# perplexity_scores - array of dictionary items with with "num_topics" and "coherence" keys
# hyperparams_list - list of hyperparameters (array of dictionaries) for LDA Model training
# dir_path - path to the directory, where models will be/are saved
# with_training - bool, if models should be trained first, or if step can be skipped (models are already exist)
def custom_grid_search(texts, corpus, id2word, hyperparams_list, coherence_cv_scores, coherence_umass_scores, perplexity_scores, dir_path, with_training):
    counter = 1


    if with_training:
        for elem in hyperparams_list:
            num_topics, alpha = elem.values()
            progress_bar(counter, len(hyperparams_list))
            counter += 1
            lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                                        id2word=id2word,
                                                        num_topics=num_topics,
                                                        workers=7,
                                                        random_state=100,
                                                        chunksize=10,
                                                        passes=10,
                                                        alpha=alpha,
                                                        iterations=50,
                                                        per_word_topics=False)

            joblib.dump(lda_model, dir_path + str(num_topics) + 'topics_' + alpha + '.jl')

    start_time = time.time()
    counter = 1
    for elem in hyperparams_list:
        num_topics, alpha = elem.values()
        progress_bar(counter, len(hyperparams_list))
        counter += 1

        lda_model = joblib.load(dir_path + str(num_topics) + 'topics_' + alpha + '.jl')

        # Coherence model to get coherence score, based on currently used corpus and dictionary
        coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, texts=texts,
                                                           dictionary=id2word, coherence='c_v')

        coherence_cv_scores.append({"num_topics": num_topics,
                                 "C_V coherence": coherence_model_lda.get_coherence()})

        # Coherence model to get coherence score, based on currently used corpus and dictionary
        coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, corpus=corpus, coherence='u_mass')

        coherence_umass_scores.append({"num_topics": num_topics,
                                 "U_Mass coherence": coherence_model_lda.get_coherence()})

        perplexity_scores.append({"num_topics": num_topics,
                                  "perplexity": lda_model.log_perplexity(corpus)})

    final_time = time.time() - start_time
    print("Total time spent: " + str(final_time) + " seconds")

    alpha = hyperparams_list[0]["alpha"]
    joblib.dump(coherence_cv_scores, dir_path + "coherence_cv_scores_" + alpha + "_alpha.jl")
    joblib.dump(coherence_umass_scores, dir_path + "coherence_umass_scores_" + alpha + "_alpha.jl")
    joblib.dump(perplexity_scores, dir_path + "perplexity_scores_" + alpha + "_alpha.jl")

def main():
    print("Started")

    # Data used for model training

    data_lemmatized = joblib.load('../../data/processed/proc_wiki_data_100MB.jl')

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]


    hyperparams_list = [{"num_topics": 4, "alpha": "symmetric"},
                        {"num_topics": 6, "alpha": "symmetric"},
                        {"num_topics": 8, "alpha": "symmetric"},
                        {"num_topics": 10, "alpha": "symmetric"},
                        {"num_topics": 12, "alpha": "symmetric"},
                        {"num_topics": 14, "alpha": "symmetric"}]

    hyperparams_list_2 = [{"num_topics": 4, "alpha": "asymmetric"},
                          {"num_topics": 6, "alpha": "asymmetric"},
                          {"num_topics": 8, "alpha": "asymmetric"},
                          {"num_topics": 10, "alpha": "asymmetric"},
                          {"num_topics": 12, "alpha": "asymmetric"},
                          {"num_topics": 14, "alpha": "asymmetric"}]

    coherence_cv_scores = []
    coherence_umass_scores = []
    perplexity_scores = []
    dir_path = "../../models/hyperparam_tuning/alpha_2/"

    custom_grid_search(data_lemmatized, corpus, id2word, hyperparams_list,
                       coherence_cv_scores, coherence_umass_scores, perplexity_scores, dir_path, True)

    print("Coherence_cv:\n" + str(coherence_cv_scores))
    print("Coherence_umass:\n" + str(coherence_umass_scores))
    print("Perplexity:\n" + str(perplexity_scores))

if __name__ == '__main__':
    main()