import csv
import spacy
import sklearn
import sklearn.metrics

from time import time
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

nlp = spacy.load('en_core_web_sm')

intent_dict_AskUbuntu = {"Make Update": 0, "Setup Printer": 1, "Shutdown Computer": 2, "Software Recommendation": 3,
                         "None": 4}
intent_dict_Chatbot = {"DepartureTime": 0, "FindConnection": 1}
intent_dict_WebApplications = {"Download Video": 0, "Change Password": 1, "None": 2, "Export Data": 3,
                               "Sync Accounts": 4,
                               "Filter Spam": 5, "Find Alternative": 6, "Delete Account": 7}

benchmark_dataset = 'AskUbuntu'  # choose from 'AskUbuntu', 'Chatbot' or 'WebApplications'


def read_CSV_datafile(filename):
    X = []
    y = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            X.append(row[0])
            if benchmark_dataset == 'AskUbuntu':
                y.append(intent_dict_AskUbuntu[row[1]])
            elif benchmark_dataset == 'Chatbot':
                y.append(intent_dict_Chatbot[row[1]])
            else:
                y.append(intent_dict_WebApplications[row[1]])
    return X, y


filename_train = "datasets/KL/" + benchmark_dataset + "/train.csv"
filename_test = "datasets/KL/" + benchmark_dataset + "/test.csv"

X_train_raw, y_train_raw = read_CSV_datafile(filename=filename_train)
X_test_raw, y_test_raw = read_CSV_datafile(filename=filename_test)


def tokenize(doc):
    """
    Returns a list of strings containing each token in `sentence`
    """
    tokens = []
    doc = nlp.tokenizer(doc)
    for token in doc:
        tokens.append(token.text)
    return tokens


def preprocess(doc):
    clean_tokens = []
    doc = nlp(doc)
    for token in doc:
        if not token.is_stop:
            clean_tokens.append(token.lemma_)
    return " ".join(clean_tokens)


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def semhash_tokenizer(text):
    tokens = text.split(" ")
    final_tokens = []
    for unhashed_token in tokens:
        hashed_token = "#{}#".format(unhashed_token)
        final_tokens += [''.join(gram)
                         for gram in list(find_ngrams(list(hashed_token), 3))]
    return final_tokens


def semhash_corpus(corpus):
    new_corpus = []
    for sentence in corpus:
        sentence = preprocess(sentence)
        tokens = semhash_tokenizer(sentence)
        new_corpus.append(" ".join(map(str, tokens)))
    return new_corpus


X_train_semhash = semhash_corpus(X_train_raw)
X_test_semhash = semhash_corpus(X_test_raw)


def get_vectorizer(corpus):
    vectorizer = CountVectorizer(ngram_range=(2, 4), analyzer='char')
    vectorizer.fit(corpus)
    return vectorizer


def benchmark(clf, X_train, y_train, X_test, y_test, target_names):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = sklearn.metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    print(sklearn.metrics.classification_report(y_test, pred,
                                                target_names=target_names))


def data_for_training(X_train_hash_data, X_test_hash_data):
    vectorizer = get_vectorizer(X_train_hash_data)

    X_train_no_HD = vectorizer.transform(X_train_hash_data).toarray()
    X_test_no_HD = vectorizer.transform(X_test_hash_data).toarray()

    return X_train_no_HD, y_train_raw, X_test_no_HD, y_test_raw


X_train, y_train, X_test, y_test = data_for_training(X_train_semhash, X_test_semhash)

target_names = ["Make Update", "Setup Printer", "Shutdown Computer", "Software Recommendation", "None"]
print('=' * 80)
print("Naive Bayes")
benchmark(MultinomialNB(alpha=.01),
          X_train, y_train, X_test, y_test, intent_dict_AskUbuntu.keys())
