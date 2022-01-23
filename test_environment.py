import joblib

import prepare, re
import zipfile
from src.visualization import visualize as vis

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
    # LDA model based on Newsgrounds articles (emails)
#    lda_model = joblib.load('models/lda_model2.jl')

    # LDA model based on Wikipedia emails
#    lda_model = joblib.load('models/lda_wiki_model.jl')
    lda_model = joblib.load('models/lda_wiki_model_10topics.jl')

    print(">>> Environment is ready to go!")

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
            if len(in_val.strip()) == 0:
                print("No text was found")
                continue
            mytext = [in_val]

            # Remove some special (unprintable) characters
            in_val = re.sub(r"[\t\r\n\v\f]+", " ", in_val)

            texts.append(in_val)

            corpora.append(prepare.get_corpus(lda_model=lda_model, text=mytext))

            # Compute Perplexity
            print('\nPerplexity: ',
                  lda_model.log_perplexity(corpora))  # a measure of how good the model is. lower the better.

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
                    print("No such file " + full_path)
                    continue

            counter = 1
            total_f = len(corpora)

            print("Corpora is processing...")

#            time0 = time.time()

#            for corpus in corpora:
#                corpus = [lda_model.id2word.doc2bow(text) for text in corpus]
#                lda_model.update(corpus)
#                progress_bar(counter, total_f)
#                counter += 1

#            total0 = time.time() - time0

#            print("Time0: " + str(total0))

            time1 = time.time()

            corpora_buf = []

            print("Updating the model, it might take a while...")

            for corpus in corpora:
                corpus = [lda_model.id2word.doc2bow(text) for text in corpus]
                corpora_buf.append(corpus[0])
                counter += 1

            # change corpora content from words to tokens
            corpora = corpora_buf

            lda_model.update(corpora)

            total1 = time.time() - time1

            print("Time spent: " + str(total1) + " seconds")

        elif (input_val == '5'):
            zip_obj = zip(corpora, texts)

            zip_list = list(zip_obj)

            if bool(corpora):
                filename = 'data/corpora/corpus_' + str(len(corpora)) + '_files.jl'
                joblib.dump(zip_list, filename)
                print("Corpora saved at " + filename)
            else:
                print("No corpora in memory, data saving failed")

        elif (input_val == '6'):
            prepare.generate_HTML_doc_LDA(ldamodel=lda_model, corpus=corpora)
            print("HTML generated at root directory")
        elif (input_val == '7'):
            break
        else:
            print("Wrong argument")

    if not bool(corpora):
        print("Corpora is empty, closing the session...")
        exit(0)

    print(">>> Visualization of the documents analysis results")
    print(">>> Choose desired visualization:\n 1) Show statistics\n 2) Show sentence chart (for 2+ texts)\n"
          " 3) t-SNE clustering\n 4) exit")

    while True:
        print("1 - Show statistics, 2 - Show sentence chart (for 2+ texts), 3 - t-SNE clustering, 4 - exit")
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
            vis.sentences_chart(lda_model=lda_model, corpus=corpora)
        elif (input_val == '3'):
            vis.t_SNE_clustering(lda_model=lda_model, corpus=corpora)
        elif (input_val == '4'):
            break
        else:
            print("Wrong argument")

    print(">>> Environment ended all tasks!")


if __name__ == '__main__':
    main()
