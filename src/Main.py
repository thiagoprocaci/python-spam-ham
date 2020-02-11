import argparse
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from src.classifier import Classifier


def main(args):
    print(args)

    if args.processText:
        data = load_initial_data()
        data = cleartext(data)
        data = createBagOfWords(data)
        data.to_csv("spam_bag.csv", sep=';', encoding='utf-8')

    data = loadProcessedData()
    create_ml_model(data)


def create_ml_model(data):
    data['y'] = data['y'].apply(pd.np.int64)

    seed = 446
    train_size = 0.8
    validation_size = 0.2


    x_columns = []
    for col in data.columns:
        if col not in ['Unnamed: 0', 'final_classifcation', 'main_text', 'id_text', 'y']:
            x_columns.append(col)

    X = data[x_columns]
    y = data['y']



    print("Splitting training set and validation set...")
    X_training, X_validation, y_training, y_validation = train_test_split(X, y,
                                                                          train_size=train_size,
                                                                          test_size=validation_size,
                                                                          random_state=seed)
    class_names = data['y'].unique().tolist()

    models = [LogisticRegression(), RandomForestClassifier(), GradientBoostingClassifier()]

    for model in models:
        print("Training classifier using " + model.__class__.__name__)
        classifier = Classifier(model)
        classifier.train(X_training, X_validation, y_training, y_validation, class_names)


def loadProcessedData():
    data = pd.read_csv("spam_bag.csv", skipinitialspace=True, sep=';')
    return data


def createBagOfWords(data, bagSize=150):
    print('Processing the text ...')
    nltk.download('punkt')

    size = len(data['main_text'])
    print("Starting tokenization process ...")
    i = 0;
    word_dict = {}
    while i < size:
        tokens = word_tokenize(data['main_text'][i])
        for token in tokens:
            if word_dict.get(token) is None:
                word_dict[token] = 0
            word_dict[token] = word_dict[token] + 1
        i += 1

    word_dict = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)

    print("Creating bag of {} words ...".format(bagSize))
    i = 0
    while i < len(word_dict) and i < bagSize:
        word = word_dict[i][0]
        data[word] = 0
        i += 1

    print("Calculating word frequency ...")
    i = 0
    while i < size:
        tokens = word_tokenize(data['main_text'][i])
        for token in tokens:
            if data.get(token) is not None:
                count = data[token][i]
                count += 1;
                data[token][i] = count
        i += 1
    return data


def cleartext(data):
    print("cleaning dataset ...")
    nltk.download('stopwords')
    size = len(data['main_text'])
    table_punctuation = str.maketrans('', '', string.punctuation)
    table_digits = str.maketrans('', '', string.digits)
    stop_words = set(stopwords.words('english'))
    i = 0
    while i < size:
        # removes punctuation
        data['main_text'][i] = data['main_text'][i].translate(table_punctuation)
        # removes numbers
        data['main_text'][i] = data['main_text'][i].translate(table_digits)
        # removes stopwords
        data['main_text'][i] = ' '.join([word for word in data['main_text'][i].split() if word not in stop_words])
        # removes words with less than 3 chars
        data['main_text'][i] = ' '.join([word for word in data['main_text'][i].split() if len(word) > 2])
        # to lower case
        data['main_text'][i] = data['main_text'][i].lower()
        #giving each line an id
        data['id_text'][i] = i
        #setting a label
        data['y'][i] = 0
        if data['final_classifcation'][i] == 'spam':
            data['y'][i] = 1
        i += 1

    return data


def load_initial_data():
    data = pd.read_csv("spam.csv", skipinitialspace=True)
    data.columns = ['final_classifcation', 'main_text', 'id_text', 'y', 'v3']
    data.drop('v3', axis=1, inplace=True)
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="Spam or Ham")
    parser.add_argument('--processText', default=True, type=lambda x: (str(x).lower() == 'true'))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
