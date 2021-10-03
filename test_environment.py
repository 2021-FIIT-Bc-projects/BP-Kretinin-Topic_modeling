import joblib

import prepare, re
import zipfile
from src.visualization import visualize as vis

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
    lda_model = joblib.load('models/lda_wiki_model.jl')

    print(">>> Environment is ready to go!")

    print(">>> Enter documets to analyze, visualized result will be shown after documents selection and analysis")
    print(">>> Documents to analyze:\n 1) Path to the document(s)\n 2) Direct input to the console\n 3) Generate HTML document with LDA topics\n 4) Close program")

    corpuses = []
    texts = []
    file_path_base = 'data/external/texts/'
    while True:
        print("Your choice (1 - path, 2 - direct input, 3 - LDA topics, 4 - exit): ")
        input_val = input()
        if (input_val == '1'):
            print("Enter name of the file located at 'data/external/texts' folder")

            file_name = str(input())
            full_path = file_path_base + file_name
            try:
                if file_name.endswith(('.txt', '.docx', '.doc')):
                    f = open(full_path, 'r')
                    print('single text file')
                elif file_name.endswith('.zip'):
                    print('zip')
                    f = zipfile.ZipFile(full_path, "r")
                else:
                    break
            except:
                print("Unable to open the file, check if file's name is correct")
                continue
            if file_name.endswith('.zip'):
                for filename in f.namelist():
                    process_file(f, filename, corpuses, texts, lda_model)
                f.close()
            elif file_name.endswith(('.txt', '.docx', '.doc')):
                process_file(f, None, corpuses, texts, lda_model)
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

            corpuses.append(prepare.get_corpus(lda_model=lda_model, text=mytext))

            # Compute Perplexity
            print('\nPerplexity: ',
                  lda_model.log_perplexity(corpuses))  # a measure of how good the model is. lower the better.

            # Used only for terminal output of the current model analysis (?)
#            if len(corpuses) > 1:
#                prepare.get_top_topic(ldamodel=lda_model, corpus=corpuses)


        elif (input_val == '3'):
            prepare.generate_HTML_doc_LDA(ldamodel=lda_model, corpus=corpuses)
            print("HTML generated at root directory")
        elif (input_val == '4'):
            break
        else:
            print("Wrong argument")

    while True:
        print("1 - Show statistics, 2 - Show sentence chart (for 2+ texts), 3 - t-SNE clustering, 4 - exit")
        input_val = input()
        if (input_val == '1'):
            vis.show_statistics(lda_model=lda_model, corpus=corpuses, texts=texts)
        elif (input_val == '2'):
            vis.sentences_chart(lda_model=lda_model, corpus=corpuses)
        elif (input_val == '3'):
            vis.t_SNE_clustering(lda_model=lda_model, corpus=corpuses)
        elif (input_val == '4'):
            break
        else:
            print("Wrong argument")

    print(">>> Environment ended all tasks!")


if __name__ == '__main__':
    main()
