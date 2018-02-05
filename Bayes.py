from collections import defaultdict, Counter
from math import log10, log
import csv


class Bayes(object):
    def __init__(self):
        self.vocab = set()

        self.class_prior = defaultdict(list)
        # key   : class
        # value : [#of doc, #n, prior]
        self.text_word = {}
        self.doc = 0
        self.prob_table = {}
        self.result_set = []  # index 0: actual class, index 1: predicted class
        self.test_doc = 0

    def read_training_datafile(self, file_name):
        with open(file_name, 'r') as fs:
            for f in fs:
                self.create_vocab(f)

    def create_vocab(self, line):
        self.doc += 1
        cls = ''
        word_count = 0
        for idx, word in enumerate(line.split()):
            if idx == 0:  # first word(class)
                cls = word
                if word in self.class_prior:
                    self.class_prior[cls][0] += 1  # counting doc in a class
                else:
                    self.class_prior[cls].append(1)  # set number of doc to zero
                    self.class_prior[cls].append(0)  # set n to zero
                    self.text_word[cls] = {}
            else:
                word_count += 1
                self.vocab.add(word)
                if word in self.text_word[cls]:
                    self.text_word[cls][word] += 1
                else:
                    self.text_word[cls][word] = 1
        self.class_prior[cls][1] += word_count  # adding n

    def read_test_datafile(self, file_name):
        with open(file_name, 'r') as tests:
            for i, t in enumerate(tests):
                l = t.split()
                self.make_prediction(set(l[1:]))
                self.result_set[i].append(l[0])
                self.test_doc += 1

    def make_prediction(self, instance):
        naive_bayes = {}
        for cls in self.class_prior.keys():
            cls_prob = 0
            for word in instance:
                cls_prob += self.search_create_prob_table(cls, word)  # big product
            naive_bayes[cls] = (self.class_prior[cls][2]) + (cls_prob)  # multiplying prior to the big product

        m = max(naive_bayes, key=naive_bayes.get)
        self.result_set.append([m])

    def search_create_prob_table(self, cls, word):
        # key : word
        # value : {class : prob}
        if word in self.prob_table:
            if cls in self.prob_table[word]:
                # retrieve
                return self.prob_table[word][cls]
            else:
                prob = self.calc_prob(cls, word)
                self.prob_table[word][cls] = prob
                return prob
        else:
            prob = self.calc_prob(cls, word)
            self.prob_table[word] = {}
            self.prob_table[word][cls] = prob
            return prob

    def calc_prob(self, cls, word):
        nk = 0
        if word in self.text_word[cls]:
            nk = self.text_word[cls][word]

        return log10((nk + 1) / ((self.class_prior[cls][1]) + len(self.vocab)))

    def cal_prior(self):
        for key, val in self.class_prior.items():
            self.class_prior[key].append((val[0] / self.doc))

    def accuracy(self):
        s = sum(i[0] == i[1] for i in self.result_set)
        print('total number of correct class:', s)
        print('Accuracy rate', s / self.test_doc)

    def do_analysis(self):
        for cls in self.result_set:
            # [# of correct class, # total # of class]
            if cls[0] not in self.analysis:
                if cls[0] == cls[1]:
                    self.analysis[cls[0]].append(1)
                    self.analysis[cls[0]].append(1)
                else:
                    self.analysis[cls[0]].append(0)
                    self.analysis[cls[0]].append(1)
            else:
                if cls[0] == cls[1]:
                    self.analysis[cls[0]][0] += 1  # count the # of correct prediction in a class
                self.analysis[cls[0]][1] += 1  # count the entire # of class in each class

    def csv(self):
        f = open('rst2.csv', 'wt')
        try:
            writer = csv.writer(f)
            writer.writerow(('result', 'class'))
            for i in self.result_set:
                writer.writerow((i[0], i[1]))
        finally:
            f.close()

b = Bayes()
import timeit

start = timeit.default_timer()
b.read_training_datafile('forumTraining.data')
b.cal_prior()
b.read_test_datafile('forumTest.data')

stop = timeit.default_timer()
print(b.accuracy())
b.csv()
print("total seconds", stop - start)
