import joblib

import prepare
from src.visualization import visualize as vis


def main():
    print(">>> Environment is preparing...")
    lda_model = joblib.load('models/lda_model2.jl')
    print(">>> Environment is ready to go!")

    print(">>> Enter documets to analyze, visualized result will be shown after documents selection and analysis")
    print(">>> Documents to analyze:\n 1) Path to the document(s)\n 2) Direct input to the console\n 3) Generate HTML document with LDA topics\n 4) Close program")
    corpuses = []
    texts = []
    while True:
        print("Your choice (1 - path, 2 - direct input, 3 - LDA topics, 4 - exit): ")
        input_val = input()
        if (input_val == '1'):
            print("1 entered, no content is here right now")
        elif (input_val == '2'):
            print("2 entered, input your text: ")
            in_val = input()
            if len(in_val.strip()) == 0:
                print("No text was found")
                continue
            mytext = [in_val]
            texts.append(mytext[0])
#            mytext = ["The elephant didn't want to talk about the person in the room."]
            corpuses.append(prepare.get_corpus(lda_model=lda_model, text=mytext))
            if len(corpuses) > 1:
                prepare.get_top_topic(ldamodel=lda_model, corpus=corpuses)



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
