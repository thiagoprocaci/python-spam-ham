from nltk.stem.snowball import SnowballStemmer
import numpy as np
import matplotlib.pyplot as plt
import re
import unicodedata
import csv
import os
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO


contractions_filename = 'conf/contractions.txt'
abbreviations_filename = 'conf/abbreviations_wikipedia_pt_and_manual_entries.txt'
stopwords_filename = "conf/stopwords.txt"


def preprocess_text(text, join_tokens=True):
    # tokens_to_delete = ['', '\n', ' ']
    #
    # text_lowered = text.lower()
    # text_preprocessed = re.split('(\W)', text_lowered)
    # text_preprocessed_final = list(filter(lambda a: a not in tokens_to_delete, text_preprocessed))

    text_tokenized_list = tokenize_text(text)
    text_preprocessed_list = preprocess_tokenized_text(text_tokenized_list, lowercase=True, remove_stopwords=False,
                                                       stemmize=False, strip_accents=True, min_word_length=False)

    if join_tokens:
        text_preprocessed_final = b' '.join(text_preprocessed_list)
    else:
        text_preprocessed_final = text_preprocessed_list

    return text_preprocessed_final

def is_number(token):
    match_idxs = re.findall("[^0-9\.,:\/\-\(\)]+", token)

    if len(match_idxs) == 0:
        token_is_number = True
    else:
        token_is_number = False

    return token_is_number


def is_email(token):
    match_idxs = re.findall("[a-z0-9]+@[a-z0-9\.]+", token)

    if len(match_idxs) > 0:
        token_is_email = True
    else:
        token_is_email = False

    return token_is_email


def find_real_periods_in_text(text, abbreviations_set):
    period_idx = text.find('.')

    while period_idx != -1:
        beginning_separator_idx = text.rfind(' ', 0, period_idx)
        end_separator_idx = text.find(' ', period_idx)

        word_to_evaluate = text[beginning_separator_idx + 1:end_separator_idx + 1].lower().strip()

        if len(word_to_evaluate) > 2:
            chars_to_remove_from_beginning_and_end = ['(', ')', ':']

            if word_to_evaluate[0] in chars_to_remove_from_beginning_and_end:
                word_to_evaluate = word_to_evaluate[1:]
            if word_to_evaluate[-1] in chars_to_remove_from_beginning_and_end:
                word_to_evaluate = word_to_evaluate[:-1]

        if word_to_evaluate not in abbreviations_set and \
            (not is_number(word_to_evaluate) or (is_number(word_to_evaluate) and period_idx == end_separator_idx - 1)) and \
                not is_email(word_to_evaluate):
            text = text[:period_idx] + ' .\n' + text[period_idx + 1:]
            period_idx += 1

        period_idx = text.find('.', period_idx + 1)

    return text


def load_words(filename, lowercase=False):
    words_set = set()

    if filename != '':
        with open(filename, 'r') as words_file:
            words_lines = words_file.readlines()
            for word in words_lines:
                if lowercase:
                    word_to_add = word.lower().strip()
                else:
                    word_to_add = word.strip()

                words_set.add(word_to_add)

    return words_set


def load_contractions(filename):
    contractions_dict = {}

    if filename != '':
        with open(filename, 'r') as contractions_file:
            contractions_lines = contractions_file.readlines()
            for contraction in contractions_lines:
                contraction_list = re.split('([\w]+) +\+ +([\w]+) += +([\w]+)', contraction)

                contraction_list_filtered = list(filter(None, contraction_list))

                contractions_dict[contraction_list_filtered[2]] = contraction_list_filtered[0:2]

    return contractions_dict


def split_text_in_sentences(text, abbreviations_set):
    text_processed = find_real_periods_in_text(text, abbreviations_set)

    sentences = re.split('\n', text_processed)

    return list(sentences)


def split_sentence_in_words(sentence, contractions_dict):
    word_list = re.split('[ ]+|(?<![0-9])([\-]+)|(?<![0-9])(:)|([\.]{2,})|([,?!;\'\"\(\)\[\]])', sentence)

    word_list_filtered = list(filter(None, word_list))

    word_list_final = []
    for word in word_list_filtered:
        if word.lower() in contractions_dict.keys():
            contraction_words = contractions_dict[word.lower()]

            if word.isupper():
                words_to_append = [contraction_word.upper() for contraction_word in contraction_words]
            elif word.istitle():
                words_to_append = [contraction_words[0].title()] + contraction_words[1:]
            else:
                words_to_append = contraction_words
        else:
            words_to_append = [word]

        word_list_final.extend(words_to_append)

    return word_list_final


def tokenize_text(text):
    abbreviations_set = load_words(abbreviations_filename, lowercase=True)
    contractions_dict = load_contractions(contractions_filename)

    tokenized_text = []

    #text_simplified = re.sub('\n(?!\n)', ' ', str(text))
    text_simplified = re.sub('\n', ' ', str(text))

    sentence_list = split_text_in_sentences(text_simplified, abbreviations_set)
    for sentence in sentence_list:
        tokenized_sentence = split_sentence_in_words(sentence, contractions_dict)

        if len(tokenized_sentence) < 1:
            continue

        tokenized_text.extend(tokenized_sentence)
        tokenized_text.extend([''])

    return tokenized_text


def preprocess_tokenized_text(token_list, lowercase=True, stemmize=True, remove_stopwords=True, strip_accents=True, min_word_length=True):
    stopwords = load_words(stopwords_filename, lowercase=True)
    stemmer = SnowballStemmer("portuguese")

    preprocessed_token_list = []

    for token in token_list:
        token_preprocessed = token

        if remove_stopwords:
            if token.lower() in stopwords:
                continue

        if min_word_length:
            if len(token_preprocessed) < 3:
                continue

        if lowercase:
            token_preprocessed = token_preprocessed.lower()

        if stemmize:
            token_preprocessed = stemmer.stem(token_preprocessed)

        if strip_accents:
            token_preprocessed = unicodedata.normalize('NFD', token_preprocessed)
            token_preprocessed = token_preprocessed.encode('ascii', 'ignore')

        preprocessed_token_list.append(token_preprocessed)

    return preprocessed_token_list


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #cax = ax.matshow(cm)
    #plt.title('Confusion matrix of the classifier')
    #fig.colorbar(cax)
    #ax.set_xticklabels([''] + classes)
    #ax.set_yticklabels([''] + classes)
    #plt.xlabel('Predicted')
    #plt.ylabel('True')
    #plt.show()

    ## As the scripts are remotely executed via console, the code
    ## below had to be commented in order not to break the execution.
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)
    #
    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")
    #
    # plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    #
    # plt.show()


def write_lines_to_csv(filename, mode, my_list):
    with open(filename, mode) as file:
        writer = csv.writer(file, delimiter=",", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        writer.writerows(my_list)

def create_dirs(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def print_parameters(**kwargs):
    for k, v in kwargs.items():
        print("%s = %s" % (k, v))


def convert_pdf_to_txt(file_object):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(file_object, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    return text
