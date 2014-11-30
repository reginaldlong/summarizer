import retrieve_data
import collections
import util
import data_format
import pickle
from collections import Counter
import random
import nltk
from optparse import OptionParser
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

def getAdditionalExamples(examples, formattedExamples, wordFrequenciesByDoc):
	additionalFeatureExamples = []
	counter = 1
	for docIndex, doc in enumerate(examples):
	    print "Adding additional features to Document ", counter
	    counter += 1

	    sentences, catchphrases = doc
	    numSentences = len(sentences)
	    numCatchphrases = len(catchphrases)

	    numSentencesPerDecile = numSentences / 10;
	    numImpt = 0

	    topCommonWords = wordFrequenciesByDoc[docIndex]

	    for i in xrange(numSentences):
	        count = formattedExamples[i]

	        additionalFeatures = {}

	        #Extract new document related features: 
	        additionalFeatures[('numSentencesFeature', 'ADD')] = numSentences
	        additionalFeatures[('numCatchphrasesFeature', 'ADD')] = numCatchphrases
	        if i == 0:
	        	additionalFeatures[('firstSentenceFeature', 'ADD')] = 1
	        else:
	        	additionalFeatures[('firstSentenceFeature', 'ADD')] = 0
	        if i == numSentences - 1:
	        	additionalFeatures[('lastSentenceFeature', 'ADD')] = 1
	        else:
	        	additionalFeatures[('lastSentenceFeature', 'ADD')] = 0

	        for j, word in enumerate(topCommonWords):
	       		featureName = "topCommonWord" + str(j + 1);
	       		if word in count:
	        		additionalFeatures[(featureName, 'ADD')] = 1
	        	else:
	        		additionalFeatures[(featureName, 'ADD')] = 0

	        if numSentencesPerDecile != 0:
	            for decile in xrange(10):
	                featureName = "decile" + str(decile + 1);
	                if i / numSentencesPerDecile == decile:
	                	additionalFeatures[(featureName, 'ADD')] = 1

	    	additionalFeatureExamples.append(additionalFeatures)

	return additionalFeatureExamples

numCatchphrasesByDoc = []
def formatExamples(examples):
    formattedExamples = []
    ytrainList = []

    #Find most common words
    wordFrequenciesByDoc = []

    counter = 1
    for sentences, catchphrases in examples:
	    print "Processing Document ", counter
	    counter += 1
	    documentWordFrequencies = Counter([])
	    numImpt = 0
	    for i, sentence in enumerate(sentences):
	        imptSentence = 0
	        for c in catchphrases:
	            if c in sentence:
	                imptSentence = 1
	                numImpt += 1

	        wordList = nltk.word_tokenize(sentence)
	        tags = nltk.pos_tag(wordList)

	        count = Counter(tags)
	        documentWordFrequencies.update(count)
	        formattedExamples.append(dict(count))
	        ytrainList.append(imptSentence)

	    numCatchphrasesByDoc.append(numImpt)
	    topCommonWords = documentWordFrequencies.most_common(10)
	    wordFrequenciesByDoc.append(topCommonWords)

    vectorizer = DictVectorizer(sparse = True)
    vectorizer.fit(formattedExamples)
    X_counts = vectorizer.transform(formattedExamples)
    y = np.asarray(ytrainList)
    tfidfTransformer = TfidfTransformer()
    X_counts = tfidfTransformer.fit_transform(X_counts)

    additionalFeatureExamples = getAdditionalExamples(examples, formattedExamples, wordFrequenciesByDoc)


    #Put them together
    #X = sp.hstack((X_counts, X_additional), format='csr')
    tfidfFormattedExamples = vectorizer.inverse_transform(X_counts)
    for i, tfidfExample in enumerate(tfidfFormattedExamples):
    	tfidfExample = tfidfExample
    	additionalFeatures = additionalFeatureExamples[i]
    	for key, value in additionalFeatures.iteritems():
    		tfidfExample[key] = value

    	tfidfFormattedExamples[i] = tfidfExample

    X = vectorizer.fit_transform(tfidfFormattedExamples)

    return X, y, vectorizer

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


def loadData():
	print "Loading examples..."
	#Each element is Doc and its catchphrases

	examples = getObjFromPklz('new_examples.pklz')
	examples = examples[0:options.numExamples]

	#Globals
	totalExampleCount = len(examples)
	print totalExampleCount, "documents loaded."

	if options.format:
		print "Formatting examples..."
		X, y, vectorizer = formatExamples(examples)

		if options.savepkl:
			print "Saving to file..."
			writeToPklz('X' + str(options.numExamples) + '.pklz', X)
			writeToPklz('y' + str(options.numExamples) + '.pklz', y)
			writeToPklz('vectorizer' + str(options.numExamples) + '.pklz', vectorizer)
	else:
		print "Getting formatting from file..."
		X = getObjFromPklz('X' + str(options.numExamples) + '.pklz')
		y = getObjFromPklz('y' + str(options.numExamples) + '.pklz')
		vectorizer = getObjFromPklz('vectorizer' + str(options.numExamples) + '.pklz')

	numCatchphrasesFeatureIndex = vectorizer.vocabulary_.get(('numCatchphrasesFeature', 'ADD'))
	firstSentenceFeatureIndex = vectorizer.vocabulary_.get(('firstSentenceFeature', 'ADD'))
	lastSentenceFeatureIndex = vectorizer.vocabulary_.get(('lastSentenceFeature', 'ADD'))

	#Split the numpy arrays into documents
	examplesByDoc = []
	yListsByDoc = []

	firstSentenceIndex = 0
	for i in xrange(X.shape[0]):
		if X[i, lastSentenceFeatureIndex] != 0:
			examplesByDoc.append(X[firstSentenceIndex:i + 1,:])
			yListsByDoc.append(np.asarray(y[firstSentenceIndex:(i + 1)]))
		if X[i, firstSentenceFeatureIndex] != 0:
			firstSentenceIndex = i

	print "Done constructing design matrix."

	return examplesByDoc, yListsByDoc


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
        confusion_matrix = testOnSet(X_test, y_test, kBest=1)
        accumulated = np.add(accumulated, confusion_matrix)

    return classifier_descr, accumulated

def runTests(examplesByDoc, yListsByDoc):
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


#Run the darn thing
parser = OptionParser()
parser.add_option('-n', action="store", dest="numExamples", type="int", default=2040, help="Number of documents to process. Default:all")
parser.add_option('--nf', action="store_false", dest="format", default=True, help="Don't reformat examples") 
parser.add_option('-s', action="store_true", dest="savepkl", default=False, help="Save formatting to pkl") 
options, remainder = parser.parse_args()
examplesByDoc, yListsByDoc = loadData()
categories = [
        'not important',
        'important'
        ]
runTests(examplesByDoc, yListsByDoc)
