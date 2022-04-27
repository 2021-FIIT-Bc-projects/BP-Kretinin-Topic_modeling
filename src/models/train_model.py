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

import joblib

import gensim.corpora as corpora
import gensim
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models



def main():
    print("Choose source of texts for learning:\n 1) Newsgrounds\n 2) Wikipedia\n 3) Reuters\n")
    in_val = input()
    if in_val == '1':

        data_lemmatized = joblib.load('../../data/processed/proc_data2.jl')

    elif in_val == '2':

        data_lemmatized = joblib.load('../../data/processed/proc_wiki_data2.jl')
#        data_lemmatized = data_lemmatized[0:5000]

    elif in_val == '3':

        data_lemmatized = joblib.load('../../data/corpora/corpus_2022-04-20_5000_files.jl')
        data_buffer = []
        for text in data_lemmatized:
            data_buffer.append(text[0][0])
        data_lemmatized = data_buffer

    else:
        print("Invalid argument")
        return -1

    print("Choose number of topics: ")
    topics_num = int(input())
    print("Chosen number of topics: " + str(topics_num))

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]

    # Build LDA model
    lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=topics_num,
                                               random_state=100,
                                               workers=7,
                                               chunksize=10,
                                               passes=20,
                                               alpha='symmetric',
                                               iterations=100,
                                               per_word_topics=False)

    pprint(lda_model.print_topics())

    print(lda_model.id2word)  # Model attributes

    # Visualize the topics
    visualisation = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(visualisation, '../../reports/LDAvis.html')

    if in_val == '1':
        joblib.dump(lda_model, '../../models/newsgrounds_' + str(topics_num) + 'topics.jl')
        # then reload it with
        lda_model = joblib.load('../../models/newsgrounds_' + str(topics_num) + 'topics.jl')
    elif in_val == '2':
        joblib.dump(lda_model, '../../models/wiki_model_' + str(topics_num) + 'topics.jl')
        # then reload it with
#        lda_model = joblib.load('../../models/big_lda_wiki_model_' + str(topics_num) + 'topics.jl')
    elif in_val == '3':
        joblib.dump(lda_model, '../../models/reuters_model_' + str(topics_num) + 'topics.jl')
        # then reload it with
    #        lda_model = joblib.load('../../models/big_lda_wiki_model_' + str(topics_num) + 'topics.jl')

    print("done")

if __name__ == '__main__':
    main()