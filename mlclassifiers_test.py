import retrieve_data
import collections
import util
import data_format
import pickle
from collections import Counter
import random
import nltk
#from optparse import OptionParser
import sys
from time import time
from sklearn.svm import LinearSVC
from sklearn import cross_validation
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction import DictVectorizer
#from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
import scipy.sparse as sp

import numpy as np
from sklearn import svm 
import gzip

numCatchphrasesByDoc = []
def formatExamples(examples):
    examples = examples[0:11]
    formattedExamples = []
    ytrainList = []
    counter = 1
    for sentences, catchphrases in examples:
        print "Document ", counter
        counter += 1
        numSentences = len(sentences)
        numCatchphrases = len(catchphrases)

        numSentencesPerDecile = numSentences / 10;
        numImpt = 0
        for i, sentence in enumerate(sentences):
            imptSentence = 0
            for c in catchphrases:
                if c in sentence:
                    imptSentence = 1
                    numImpt += 1

            wordList = sentence.split(' ');
            count = dict(Counter(nltk.word_tokenize(wordList)))

            #Extract new document related features: 
            count['numSentencesFeature'] = numSentences
            count['numCatchphrasesFeature'] = numCatchphrases
            count['firstSentenceFeature'] = 50 * (i==0)
            count['lastSentenceFeature'] = 50 * (i==(numSentences-1))

            if numSentencesPerDecile != 0:
                for decile in xrange(10):
                    featureName = "decile" + str(i + 1);
                    count[featureName] = (i / numSentencesPerDecile == decile)

            formattedExamples.append(count)
            ytrainList.append(imptSentence)
        numCatchphrasesByDoc.append(numImpt)

    return formattedExamples, ytrainList

def getObjFromPklz(infilename):
    f = gzip.open(infilename, 'rb')
    try:
        return pickle.load(f)
    finally:
        f.close()

def writeToPklz(outfilename, obj):
    output = gzip.open(outfilename, 'wb')
    try:
        pickle.dump(obj, output, -1)
    finally:
        output.close()

'''infilename = 'sentences.pklz'
f = gzip.open(infilename, 'rb')
try:
    sentences = pickle.load(f)
finally:
    f.close()
infilename = 'catchphrases.pklz'
f = gzip.open(infilename, 'rb')
try:
    catchphrases = pickle.load(f)
finally:
    f.close()'''

print "Loading examples..."
#Each element is Doc and its catchphrases

examples = getObjFromPklz('new_examples.pklz')
#Globals
totalExampleCount = len(examples)
print totalExampleCount, " examples loaded."

allDict, yList = formatExamples(examples)

#writeToPklz('features.pklz', allDict)
#writeToPklz('valueList.pklz', yList)
#allDict = getObjFromPklz('features.pklz')
#yList = getObjFromPklz('valueList.pklz')
vectorizer = DictVectorizer(sparse = True)
vectorizer.fit(allDict)

X = vectorizer.transform(allDict)
y = np.asarray(yList)

#Tfidf
tfidfTransformer = TfidfTransformer()
X = tfidfTransformer.fit_transform(X, y)
#map from indices to feature names
feature_names = np.asarray(vectorizer.get_feature_names())


numCatchphrasesFeatureIndex = vectorizer.vocabulary_.get('numCatchphrasesFeature')
firstSentenceFeatureIndex = vectorizer.vocabulary_.get('firstSentenceFeature')
lastSentenceFeatureIndex = vectorizer.vocabulary_.get('lastSentenceFeature')

#Split the numpy arrays into documents
examplesByDoc = []
yListsByDoc = []

firstSentenceIndex = 0
for i in xrange(X.shape[0]):
    if X[i, lastSentenceFeatureIndex] != 0:
        examplesByDoc.append(X[firstSentenceIndex:i + 1, :])
        yListsByDoc.append(y[firstSentenceIndex:(i + 1)])
        firstSentenceIndex = i + 1

categories = [
        'not important',
        'important'
        ]
print "Done constructing design matrix."

def predict(classifier, X):
    return classifier.predict(X)

def train(classifer, X, y):
    classifier.fit(X, y)

def benchmark(classifier, X_train, y_train, testTuples):
    classifier_descr = str(classifier).split('(')[0]
    def testOnSet(X, y, kBest=-1):
        #Prediction
        pred = predict(classifier, X)

        if classifier_descr == "LinearSVC":
            predicted_test_scores= classifier.decision_function(X) 

            if kBest != -1:
                kBest = y.tolist().count(1)
                best_ind_array = np.argsort(-predicted_test_scores)[:kBest]
                pred = np.zeros(pred.size, dtype = int)
                for bestIndex in best_ind_array:
                    pred[bestIndex] = 1
        elif classifier_descr != "NearestCentroid":
            if kBest != -1:
                kBest = y.tolist().count(1)
                predicted_test_scores = classifier.predict_proba(X)
                best_ind_array = np.argsort(-predicted_test_scores[:,1])[:kBest]
                pred = np.zeros(pred.size, dtype = int)
                for bestIndex in best_ind_array:
                    pred[bestIndex] = 1


        score = metrics.f1_score(y, pred)
        print("f1-score:   %0.3f" % score)

        #Precision, recall for each class
        print("classification report:")
        print(metrics.classification_report(y, pred, target_names=categories))

        #Number of correct positives, false pos, false neg, correct neg
        print("confusion matrix:")
        confusion_matrix =  metrics.confusion_matrix(y, pred)
        print(confusion_matrix)
        return confusion_matrix

    print('_' * 80)
    print("Training: ")
    print(classifier)
    classifier.fit(X_train, y_train)

    print('_' * 40)
    print("Testing on Training Set: ")
    testOnSet(X_train, y_train)
    print("Testing on Test Set: ")	

    accumulated = np.array([[0,0],[0,0]])

    for X_test, y_test, docIndex in testTuples:
        confusion_matrix = testOnSet(X_test, y_test, kBest=numCatchphrasesByDoc[docIndex])
        accumulated = np.add(accumulated, confusion_matrix)

    return classifier_descr, accumulated


def format():
    infilename = 'sentences.pklz'
    f = gzip.open(infilename, 'rb')
    try:
        sentences = pickle.load(f)
    finally:
        f.close()
    infilename = 'catchphrases.pklz'
    f = gzip.open(infilename, 'rb')
    try:
        catchphrases = pickle.load(f)
    finally:
        f.close()
    infilename = 'new_examples.pklz'
    f = gzip.open(infilename, 'rb')
    try:
        examples = pickle.load(f)
    finally:
        f.close()

    #Globals
    totalExampleCount = len(examples)

    allDict, yList = formatExamples(examples)
    vectorizer = DictVectorizer(sparse = True)
    vectorizer.fit(allDict)

    X = vectorizer.transform(allDict)
    y = np.asarray(yList)

    tfidfTransformer = TfidfTransformer()
    X = tfidfTransformer.fit_transform(X, y)

    #map from indices to feature names
    feature_names = np.asarray(vectorizer.get_feature_names())



def runTests():
    #This is run globally
    #X_train, y_train, X_test, y_test = format()

    print "Running Tests..."
    results = []

    #10-fold cross validation
    kf = cross_validation.KFold(len(examplesByDoc), n_folds=10)	

    for train_index, test_index in kf:

        #We pick 90% of documents to train on, 10% to test on each fold 

        #build k-fold partitions by documents
        testTuples = []
        for i in test_index:
            testTuples.append((examplesByDoc[i], yListsByDoc[i], i))

        first = True
        for i in train_index:

            if first:
                X_train = examplesByDoc[i]
                y_train = yListsByDoc[i]
                first = False
            else:
                X_train = sp.vstack((X_train, examplesByDoc[i]), format='csr')
                y_train = np.concatenate((y_train, yListsByDoc[i]))

        # Train Liblinear model with L1, L2 regularization
        for penalty in ["l2"]:
            print('=' * 80)
            print("%s regularization" % penalty.upper())
            classifier = LinearSVC(loss='l2', penalty=penalty, dual=False, C = 1, class_weight='auto')
            results.append(benchmark(classifier, X_train, y_train, testTuples))


        # Train sparse Naive Bayes classifiers
        print('=' * 80)
        print("Naive Bayes")
        classifier = MultinomialNB(alpha=.01)
        results.append(benchmark(classifier, X_train, y_train, testTuples))
        classifier = BernoulliNB(alpha=.01)
        results.append(benchmark(classifier, X_train, y_train, testTuples))

    print('='*80)
    print 'Aggregate'
    aggregate_results = {}
    for classifier_descr, confusion_matrix in results:
        accumulated = aggregate_results.get(classifier_descr, np.array([[0,0],[0,0]]))
        aggregate_results[classifier_descr] = np.add(accumulated, confusion_matrix)

    for classifier_descr, confusion_matrix in aggregate_results.iteritems():
        print('-'*40)
        print classifier_descr
        print confusion_matrix
        recall = confusion_matrix[1,1] / float(confusion_matrix[1,1] + confusion_matrix[1,0])
        precision = confusion_matrix[1,1] / float(confusion_matrix[1,1] + confusion_matrix[0,1])
        print "Precision: ", precision, " Recall: ", recall

runTests()
