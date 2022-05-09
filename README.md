BP-Kretinin-Topic_modeling
==============================

Mykyta's BP project FIIT STU Bratislava, Topic modeling 

Registration number of work in the information system
------------

FIIT-5212-105326

BP Task
------------

Viac ako 80% dát, ktoré sú dnes k dispozícii, sú neštruktúrované dáta. Sú kľúčom k cenným informáciám. Na spracovanie a interpretáciu textových údajov používame rôzne techniky z oblasti spracovania prirodzeného jazyka. Jedna z nich je tematické modelovanie pomocou LDA (pozn. Latent Dirichlet Allocation je metóda strojového učenia bez učiteľa). Analyzujte súčasný stav problematiky v oblasti spracovania textu a jazyka pomocou moderných knižníc. Implementuje základný modul na zistenie najdiskutovanejšej témy vo väčšom množstve dokumentov. Výsledok vyhodnoťte pomocou dostupných metrík. Výsledok LDA modelu vizualizujte vhodným spôsobom.

Project Organization
------------


    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── corpora        <- Processed corpora ready for use.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models (.jl extension),
    │                         model predictions, or model summaries
    │
    ├── reports            <- Generated diagrams and graphs (as HTML pages)
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pipreqs`
    │
    ├── test_environment.py    <- environment for topic modeling (run this one!)
    │
    ├── prepare.py         <- functions for input data preprocessing and basic analysis
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   │
    │   │   ├── make_dataset.py         <- create a dataset from newsgrouns articles 
    │   │   │                              (currently unused, has been used in functionality testing)
    │   │   │                              
    │   │   └── make_wiki_dataset.py    <- create a wiki dataset from a Wikipedia
    │   │                                  articles dump (defined in the code)
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   │
    │   │   ├── hyperparameters_tuning.py    <- hyperparameter tuning experiment, where
    │   │   │                                   multiple models are trained and compared
    │   │   │
    │   │   └── train_model.py               <- LDA (multicore) model creation using a chosen dataset
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py

Installation
--------

1) Clone the project from the Git repository
2) run `pip install -r requirements.txt` in the root folder of the project
3) run `python -m nltk.downloader stopwords`
4) run `python -m spacy download en`
5) run `pip install html5lib`
6) (extra step, program will work without it) run `pip install python-Levenshtein` (Microsoft Visual c++ 14 Build tools are required)

User Manual
--------

**Notice:** For testing there are trained models (/models folder), so user can start from **Step 2**.

1) First, user has to train LDA model, what can be done by running ***train_model.py***. If user wants, he can manually set/change predefined parameters of the model in the code. Also, user has to provide a training dataset, which will be used in model training.
2) After model has been trained, user is able to do the tests. For these purposes, there are (git repository) few text datasets - scrapped Reuters articles. They are located at */data/external/texts*.
3) So, user has to launch (run) ***test_environment.py***, where he will be requested to choose model, and then - to choose one of the options: 
    1 - path,
    2 - direct input,
    3 - Crawl for last N articles at reuters.com,
    4 - use corpora,
    5 - save corpora,
    6 - Intertopic Distance Map,
    7 - exit
4) Therefore, as the main scenario, will be chosen option "**1**", and user will enter the name of the .zip archive (our testing dataset)
5) After being processed, its corpora may be saved (optional, option "***5***") and has to be applied to update the model (option "***4***").
6) After that, user can go to the next part of the testing - data visualization (option "***7***" - exit).
7) User will see the distribution of documents from the dataset among LDA model topics. Then next data may be visualized: statistics about documents in the dataset (option "***1***"),  t-SNE clustering (option "***2***"), coherence and perplexity scores (option "***3***").
8) When all tests were done, user will close the program (option "***4***" -  exit)

Code files providing a functionality
--------

1) ***test_environment.py*** - main program, where tests are executed. Requires at least 1 trained LDA model (.jl extension) in \models directory to work
2) ***make_wiki_dataset.py*** - dataset creation from the Wikipedia dump
                          (dump shall be stored in `gensim.test.utils.datapath()` folder, which is usually
                           **C:\Users\\\<User>\PycharmProjects\\\<pythonProject>\venv\Lib\site-packages\gensim\test\test_data** in PyCharm IDE case)
4) ***train_model.py*** - LDA model creation and training
5) ***hyperparameters_tuning.py*** - execution of hyperparameter tuning experiment, to define the best hyperparameters for a given training dataset

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
