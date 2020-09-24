import csv
import spacy
import sklearn
import sklearn.metrics

from time import time

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import xgboost as xgb


class SemHashClassifier:
    def __init__(self, intent_dict: dict):
        self.intent_dict = intent_dict
        self.id2intent = {v: k for k, v in intent_dict.items()}
        self.nlp = spacy.load('en_core_web_sm')
        self.vectorizer = CountVectorizer(analyzer='word')

        # self.vectorizer = TfidfVectorizer(analyzer='word')

    def tokenize(self, doc):
        """
        Returns a list of strings containing each token in `sentence`
        """
        tokens = []
        doc = self.nlp.tokenizer(doc)
        for token in doc:
            tokens.append(token.text)
        return tokens

    def split(self, doc):
        clean_tokens = []
        doc = self.nlp(doc)
        for token in doc:
            if not token.is_stop:
                clean_tokens.append(token.lemma_)
        return " ".join(clean_tokens)

    @staticmethod
    def find_ngrams(input_list, n):
        return zip(*[input_list[i:] for i in range(n)])

    def semhash_tokenizer(self, text):
        tokens = text.split(" ")
        final_tokens = []
        for unhashed_token in tokens:
            hashed_token = "#{}#".format(unhashed_token)
            final_tokens += [''.join(gram)
                             for gram in list(self.find_ngrams(list(hashed_token), 3))]
        return final_tokens

    def semhash_corpus(self, corpus):
        new_corpus = []
        for sentence in corpus:
            sentence = self.split(sentence)
            tokens = self.semhash_tokenizer(sentence)
            new_corpus.append(" ".join(map(str, tokens)))
        return new_corpus

    def preprocess(self, X_train):
        X_train_semhash = self.semhash_corpus(X_train)
        return X_train_semhash

    def fit(self, X_train, y_train):
        X_train_semhash = self.preprocess(X_train)
        self.vectorizer.fit(X_train_semhash)
        train_feat = self.vectorizer.transform(X_train_semhash).toarray()
        # self.clf = MultinomialNB(alpha=.01)
        # self.clf = xgb.XGBClassifier(n_estimators=80, max_depth=1, subsample=0.8)
        # self.clf = RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
        #         max_iter=None, normalize=False, random_state=None,
        #         solver='lsqr', tol=0.01)
        # self.clf = LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
        #                      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
        #                      multi_class='ovr', penalty='l2', random_state=None, tol=0.001,
        #                      verbose=0)
        # self.clf = LinearSVC()

        svm = LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                        intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                        multi_class='ovr', penalty='l2', random_state=None, tol=0.001,
                        verbose=0)
        self.clf = CalibratedClassifierCV(svm)

        t0 = time()
        self.clf.fit(train_feat, y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)
        pred = self.clf.predict(train_feat)

        score = sklearn.metrics.accuracy_score(y_train, pred)
        print("train accuracy:   %0.3f" % score)

    def evaluate(self, X_test, y_test):
        semhash_feat = self.preprocess(X_test)
        final_feat = self.vectorizer.transform(semhash_feat).toarray()
        t0 = time()
        pred = self.clf.predict(final_feat)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = sklearn.metrics.accuracy_score(y_test, pred)
        print("test accuracy:   %0.3f" % score)

        print(sklearn.metrics.confusion_matrix(y_test, pred))

        print(sklearn.metrics.classification_report(y_test, pred, target_names=self.intent_dict.keys()))

    def predict(self, text):
        semhash_feat = self.preprocess([text])
        final_feat = self.vectorizer.transform(semhash_feat).toarray()
        pred = self.clf.predict(final_feat)[0]
        return {"intent_id": pred, "intent": self.id2intent[pred]}

    def predict_proba(self, text):
        semhash_feat = self.preprocess([text])
        final_feat = self.vectorizer.transform(semhash_feat).toarray()
        pred = self.clf.predict_proba(final_feat)[0]
        return [{"intent": self.id2intent[i], "prob": prob} for i, prob in enumerate(pred)]


if __name__ == '__main__':
    intent_dict_AskUbuntu = {"Make Update": 0, "Setup Printer": 1, "Shutdown Computer": 2, "Software Recommendation": 3,
                             "None": 4}
    intent_dict_Chatbot = {"DepartureTime": 0, "FindConnection": 1}
    intent_dict_WebApplications = {"Download Video": 0, "Change Password": 1, "None": 2, "Export Data": 3,
                                   "Sync Accounts": 4,
                                   "Filter Spam": 5, "Find Alternative": 6, "Delete Account": 7}

    benchmark_dataset = 'AskUbuntu'  # choose from 'AskUbuntu', 'Chatbot' or 'WebApplication'


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

    # model = SemHashClassifier(intent_dict_WebApplications)
    model = SemHashClassifier(intent_dict_AskUbuntu)

    model.fit(X_train_raw, y_train_raw)

    model.evaluate(X_test_raw, y_test_raw)

    # print(model.predict("Which map will you suggest?"))
    # print(model.predict("can you give me some advice on which map to use"))

    print(model.predict_proba("Which map will you suggest?"))
    print(model.predict_proba("can you give me some advice on which map to use"))
