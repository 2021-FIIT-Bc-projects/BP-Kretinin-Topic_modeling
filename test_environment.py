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

from gensim.models.ldamodel import LdaModel

import prepare
import re
import zipfile
from src.visualization import visualize as vis
from datetime import date
from os import listdir
from os.path import isfile, join, basename

import time

import warnings
warnings.filterwarnings("ignore")

def progress_bar(iteration, total):
    total_len = 100
    percent_part = ("{0:.2f}").format(100 * (iteration / total))
    filled = int(total_len * iteration / total)
    bar = '█' * filled + '-' * (total_len - filled)
    print(f'\r Progress: [{bar}] {percent_part}%', end='')
    if iteration == total:
        print()

def process_file(file, filename, corpuses, texts, lda_model):
    text = file.read(filename)
    if type(text) == bytes:
        text = text.decode('utf-8')

    if len(text.strip()) == 0:
        print("No text was found")
        return
    mytext = [text]

    # Remove some special (unprintable) characters
    text = re.sub(r"[\t\r\n\v\f]+", " ", text)

    texts.append(text)

    corpuses.append(prepare.get_corpus(lda_model=lda_model, text=mytext))

def main():

    print(">>> Environment is preparing...")

    models_path = "models/"

    print(">>> Environment is ready to go!")

    models_list = [f for f in listdir(models_path) if isfile(join(models_path, f))]
    models_names = []
    for model in models_list:
        name = basename(model)
        if re.search("\.jl$", name):
            models_names.append(name)

    if models_names:
        print(">>> Choose one of the following models from /models folder: ")
        for model in models_names:
            print(model)
        print(">>> Name of the model: ")
        model_name = str(input())
    else:
        print(">>> Error! No models found to do tests")
        return 0

    lda_model = joblib.load(models_path + model_name)

    if bool(model_name):
        print(">>> Using trained model: " + model_name)

    print(">>> Enter documents to analyze, visualized result will be shown after documents selection and analysis")
    print(">>> Documents to analyze and other options:\n 1) Path to the document(s)\n 2) Direct input to the console\n"
          " 3) Crawl for last N articles at reuters.com\n"
          " 4) Use current corpora (of this session) or use one from the memory\n"
          " 5) Save the corpora of the current session\n 6) Generate HTML document with LDA topics of the model\n"
          " 7) Close program")

    corpora = []
    texts = []

    while True:
        print("Your choice (1 - path, 2 - direct input, 3 - Crawl for last N articles at reuters.com, 4 - use corpora,"
              " 5 - save corpora, 6 - Intertopic Distance Map, 7 - exit): ")
        input_val = input()
        if (input_val == '1'):
            file_path_base = 'data/external/texts/'
            print("Enter the name of the file located at 'data/external/texts' folder")

            file_name = str(input())
            full_path = file_path_base + file_name
            try:
                if file_name.endswith(('.txt', '.docx', '.doc')):
                    f = open(full_path, 'r')
                    print('single text file is processing...')
                elif file_name.endswith('.zip'):
                    print('zip archive is processing...')
                    f = zipfile.ZipFile(full_path, "r")
                else:
                    break
            except:
                print("Unable to open the file, check if file's name is correct")
                continue
            if file_name.endswith('.zip'):
                total_f = len(f.namelist())
                counter = 1
                for filename in f.namelist():
                    progress_bar(counter, total_f)
                    counter += 1
                    process_file(f, filename, corpora, texts, lda_model)
                f.close()

            elif file_name.endswith(('.txt', '.docx', '.doc')):
                process_file(f, None, corpora, texts, lda_model)
                f.close()
        elif (input_val == '2'):
            print("2 entered, input your text: ")
            in_val = input()
            user_text = ""
            while in_val != '':     # get all lines from input, if it contains newline characters
                user_text += in_val
                user_text += " "
                in_val = input()
            if len(user_text.strip()) == 0:
                print("No text was found")
                continue
            mytext = [user_text]

            # Remove some special (unprintable) characters
            user_text = re.sub(r"[\t\r\n\v\f]+", " ", user_text)

            texts.append(user_text)

            corpora.append(prepare.get_corpus(lda_model=lda_model, text=mytext))

            # Compute Perplexity
            #print('\nPerplexity: ',
            #      lda_model.log_perplexity(corpora))  # a measure of how good the model is. lower the better.

            # Used only for terminal output of the current model analysis (?)
#            if len(corpuses) > 1:
#                prepare.get_top_topic(ldamodel=lda_model, corpus=corpuses)

        elif (input_val == '3'):
            print("Input number of articles to crawl: ")
            num = int(input())
            if type(num) != int:
                print("Wrong input format")
                continue
            prepare.get_last_N_articles_from_reuters(num)
            print("Done")

        elif (input_val == '4'):
            #
            if not bool(corpora):
                try:
                    file_path_base = 'data/corpora/'
                    print("Enter the name of the file located at 'data/corpora' folder")

                    corpora_name = str(input())
                    full_path = file_path_base + corpora_name

                    # read corpora dump as zipped list and separate preprocessed text
                    # (array of words) from documents' texts
                    zip_list = joblib.load(full_path)
                    zip_obj = zip(*zip_list)
                    unzip_list = list(zip_obj)
                    corpora = unzip_list[0]
                    texts = unzip_list[1]
                except:
                    print("error")
                    continue

            print("Corpora is processing...")

            time1 = time.time()

            corpora_buf = []

            print("Updating the model, it might take a while...")

            for corpus in corpora:
                corpus = [lda_model.id2word.doc2bow(text) for text in corpus]
                corpora_buf.append(corpus[0])

            # change corpora content from words to tokens
            corpora = corpora_buf

            lda_model.update(corpora)

            total1 = time.time() - time1

            print("Time spent: " + str(total1) + " seconds")

            # Uncomment to save an updated model, to be used later in the "8-th" testing only option later
            # lda_model.save("models/interim/" + model_name + "_" + str(len(corpora)) + ".jl")
        elif (input_val == '5'):
            zip_obj = zip(corpora, texts)

            zip_list = list(zip_obj)

            if bool(corpora):
                filename = 'data/corpora/corpus_' + str(date.today()) + '_' + str(len(corpora)) + '_files.jl'
                joblib.dump(zip_list, filename)
                print("Corpora saved at " + filename)
            else:
                print("No corpora in memory, data saving failed")

        elif (input_val == '6'):
            prepare.generate_HTML_doc_LDA(ldamodel=lda_model, corpus=corpora)
            print("HTML generated at /reports directory")
        elif (input_val == '7'):
            break
        elif (input_val == '8'):
            print("TESTING ONLY")
            print("This option uses already updated model with the corpus used in testing")
            print("After model and corpus selection go straight to the next step")
            print("Model name:")
            model_name = input()
#            lda_model = joblib.load("models/interim/" + model_name)
            lda_model = LdaModel.load("models/interim/" + model_name, mmap='r')

            file_path_base = 'data/corpora/'
            print("Enter the name of the file located at 'data/corpora' folder")

            corpora_name = str(input())
            full_path = file_path_base + corpora_name

            # read corpora dump as zipped list and separate preprocessed text
            # (array of words) from documents' texts
            zip_list = joblib.load(full_path)
            zip_obj = zip(*zip_list)
            unzip_list = list(zip_obj)
            corpora = unzip_list[0]
            texts = unzip_list[1]

            corpora_buf = []
            for corpus in corpora:
                corpus = [lda_model.id2word.doc2bow(text) for text in corpus]
                corpora_buf.append(corpus[0])

            # change corpora content from words to tokens
            corpora = corpora_buf

        else:
            print("Wrong argument")

    if not bool(corpora):
        print("Corpora is empty, closing the session...")
        exit(0)
    try:
        vis.show_docs_per_topic(lda_model, corpora)
    except:
        print("Not enough information to get analysis results")
        print("Probably, selected corpora hasn't been applied (option 4 - use corpora)")
        return 0

    print(">>> Visualization of the documents analysis results")
    print(">>> Choose desired visualization:\n 1) Show statistics\n "
          "2) t-SNE clustering\n 3) Calculate coherence and perplexity\n 4) exit")

    while True:
        print("1 - Show statistics, 2 - t-SNE clustering,"
              " 3 - Coherence and Perplexity, 4 - exit")
        input_val = input()
        if (input_val == '1'):
            print("Choose documents number to be displayed (up to 15, 0 for the first ten)")
            num = int(input("Number of documents to be shown in the statistics table: "))
            text_numbers = []

            if (num > 0):
                print("Documents' numbers : ")
                for i in range(0, num):
                    text_numbers.append(int(input()))

            vis.show_statistics(lda_model=lda_model, corpus=corpora, texts=texts, text_numbers=text_numbers)
        elif (input_val == '2'):
            vis.t_SNE_clustering(lda_model=lda_model, corpus=corpora)
        elif (input_val == '3'):
            vis.show_coherence_and_perplexity(model=lda_model, corpus=corpora)
        elif (input_val == '4'):
            break
        else:
            print("Wrong argument")

    print(">>> Environment ended all tasks!")


if __name__ == '__main__':
    main()
