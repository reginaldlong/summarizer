import retrieve_data
import processExamples
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
from sklearn.linear_model import SGDClassifier
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
import heapq as heapq

def getAdditionalExamples(examples, formattedExamples, wordFrequenciesByDoc):
    additionalFeatureExamples = []
    counter = 1
    sentenceCounter = 0
    for docIndex, example in enumerate(examples):
        print "Adding additional features to Document ", counter
        counter += 1
        if fileNameVariance == "with_title" or fileNameVariance == "oracle_with_title":
            sentences, catchphrases, title = example
        else:
            sentences, catchphrases = example

        numSentences = len(sentences)
        numCatchphrases = len(catchphrases)

        numSentencesPerDecile = numSentences / 10;
        numImpt = 0

        topCommonWords = wordFrequenciesByDoc[docIndex]

        for i in xrange(numSentences):
            count = formattedExamples[sentenceCounter]

            sentenceCounter += 1
            additionalFeatures = {}
            #Extract document context features: 

            sentence = sentences[i]
            #add containing title feature
            numTitleWordsPresent = 0
            titleWordList = title.split(" ")
            for titleWord in titleWordList:
                if titleWord in sentence:
                    numTitleWordsPresent += 1
            additionalFeatures[('numTitleWords', 'ADD')] = numTitleWordsPresent


            #number of sentences
            additionalFeatures[('numSentencesFeature', 'ADD')] = numSentences
            
            #Sentence position
            if i == 0:
            	additionalFeatures[('firstSentenceFeature', 'ADD')] = 1
            else:
            	additionalFeatures[('firstSentenceFeature', 'ADD')] = 0
            if i == numSentences - 1:
            	additionalFeatures[('lastSentenceFeature', 'ADD')] = 1
            else:
            	additionalFeatures[('lastSentenceFeature', 'ADD')] = 0

            #Presence of high frequency word
            for j, word in enumerate(topCommonWords):
                featureName = "topCommonWord" + str(j + 1);
                if word in count:
                    additionalFeatures[(featureName, 'ADD')] = 1
                else:
                    additionalFeatures[(featureName, 'ADD')] = 0

            #Relative sentence position by tenths
            if numSentencesPerDecile != 0:
                for decile in xrange(10):
                    featureName = "decile" + str(decile + 1);
                    if i / numSentencesPerDecile == decile:
                        additionalFeatures[(featureName, 'ADD')] = 1
            additionalFeatureExamples.append(additionalFeatures)

    return additionalFeatureExamples

firstOrLastSentence = []
def formatExamples(examples):
    formattedExamples = []
    ytrainList = []

    #Find most common words
    wordFrequenciesByDoc = []

    #For document in documents
    for counter, example in enumerate(examples):
        
        if fileNameVariance == "with_title" or fileNameVariance == "oracle_with_title":
            sentences, catchphrases, title = example
        else:
            sentences, catchphrases = example
        
        print "Processing Document " + str(counter + 1)
        documentWordFrequencies = Counter([])
        numSentences = len(sentences)
        
        #importance labels for this document
        labels = [0 for i in xrange(numSentences)]

        #Holds max heaps of catchphrases for each catchphrase
        catchphraseSimilarityRanking = [[] for i in xrange(len(catchphrases))]

        #Label importance if a sentence has the highest similarity score 
        #to any catchphrase out of all sentences in the document. Thus there is one
        #most similar sentence for every catchphrase, so numImptSentences = numCatchphrases
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()

            if i == 0:
                firstOrLastSentence.append(1)
            elif i == numSentences - 1:
                firstOrLastSentence.append(2)
            else: 
                firstOrLastSentence.append(0)

            if fileNameVariance != "baseline":
                wordList = sentence.split(" ")
                wordFreq = Counter(wordList)
                
                #determine similarity score to each catchphrase
                for catchphraseIndex, c in enumerate(catchphrases):
                    catchphraseWordList = c.strip().split(" ")
                    catchphraseWordList = Counter(catchphraseWordList)
                    
                    similarityScore = 0
                    for catchphraseWord, freq in catchphraseWordList.iteritems():
                        similarityScore += wordFreq.get(catchphraseWord, 0) * freq
                    
                    #store negative score in min heap for max heap
                    heapq.heappush(catchphraseSimilarityRanking[catchphraseIndex], (-similarityScore, i)) #Store sentence index also to recover it later         
                
                
                tagTuples = nltk.pos_tag(wordList)

                numWordsPerThird = len(sentence) / 3
                #Add tags themselves
                tags = []
                tagPos = []
                for j, tup in enumerate(tagTuples):
                    word, tag = tup
                    tags.append((tag, 'ADD'))
                    if numWordsPerThird != 0:
                        tagPositionFeatureName = 'tagPos' + str(j / numWordsPerThird)
                        tagPos.append((tagPositionFeatureName, tag))

                
                count = Counter(tagTuples)
                documentWordFrequencies.update(count)
                count.update(Counter(tags))
                count.update(Counter(tagPos))
                
            else:
                #baseline is in already split format, unfortunately
                joinedSentence = " ".join(sentence)
                imptSentence = 0
                for c in catchphrases:
                    if c in joinedSentence:
                        imptSentence = 1
                count = Counter(sentence)
            
            formattedExamples.append(dict(count))
        
        #Determine important sentences
        for catchphraseHeap in catchphraseSimilarityRanking:
            score, sentenceIndex = heapq.heappop(catchphraseHeap)
            labels[sentenceIndex] = 1
            
        
        #update global label list
        ytrainList = ytrainList + labels

        if fileNameVariance != "baseline":
            topCommonWords = documentWordFrequencies.most_common(10)
            wordFrequenciesByDoc.append(topCommonWords)

    vectorizer = DictVectorizer(sparse = True)
    vectorizer.fit(formattedExamples)
    X = vectorizer.transform(formattedExamples)
    y = np.asarray(ytrainList)
    
    if fileNameVariance != "baseline":
        tfidfTransformer = TfidfTransformer()
        X = tfidfTransformer.fit_transform(X)

        additionalFeatureExamples = getAdditionalExamples(examples, formattedExamples, wordFrequenciesByDoc)


        #Put them together
        tfidfFormattedExamples = vectorizer.inverse_transform(X)
        for i, tfidfExample in enumerate(tfidfFormattedExamples):
            additionalFeatures = additionalFeatureExamples[i]
            for key, value in additionalFeatures.iteritems():
    		tfidfExample[key] = value
                tfidfFormattedExamples[i] = tfidfExample

        X = vectorizer.fit_transform(tfidfFormattedExamples)

    return X, y, vectorizer



def loadData():
    examplesFileName = "new_examples_with_title.pklz"
    if options.useOracle:
        examplesFileName = "oracle_examples_with_title.pklz"
    print "Loading examples from " + examplesFileName + "..."
    #Each element is Doc and its catchphrases
    
    examples = processExamples.getObjFromPklz(examplesFileName)
    if not options.useOracle:
        examples = examples[0:options.numExamples]

    #Globals
    totalExampleCount = len(examples)
    print totalExampleCount, "documents loaded."

    if options.format:
        print "Formatting examples..."
        X, y, vectorizer = formatExamples(examples)

        if options.savepkl:
            print "Saving to file..."
            processExamples.writeToPklz('X' + str(options.numExamples) + fileNameVariance + '.pklz', X)
            processExamples.writeToPklz('y' + str(options.numExamples) +  fileNameVariance +  '.pklz', y)
            processExamples.writeToPklz('vectorizer' + str(options.numExamples) + fileNameVariance + '.pklz', vectorizer)
    else:
        if options.useOracle:
            print "Getting formatting from " + '__' + "oracle_" + fileNameVariance + '.pklz'
            X = processExamples.getObjFromPklz('X' + "oracle_" + fileNameVariance + '.pklz')
            y = processExamples.getObjFromPklz('y' + "oracle_" + fileNameVariance + '.pklz')
            vectorizer = processExamples.getObjFromPklz('vectorizer' + "oracle_" + fileNameVariance + '.pklz')
        else:
            print "Getting formatting from " + '__' + str(options.numExamples) + fileNameVariance + '.pklz'
            X = processExamples.getObjFromPklz('X' + str(options.numExamples) + fileNameVariance + '.pklz')
            y = processExamples.getObjFromPklz('y' + str(options.numExamples) + fileNameVariance + '.pklz')
            vectorizer = processExamples.getObjFromPklz('vectorizer' + str(options.numExamples) + fileNameVariance + '.pklz')

    numCatchphrasesFeatureIndex = vectorizer.vocabulary_.get(('numCatchphrasesFeature', 'ADD'))
    firstSentenceFeatureIndex = vectorizer.vocabulary_.get(('firstSentenceFeature', 'ADD'))
    lastSentenceFeatureIndex = vectorizer.vocabulary_.get(('lastSentenceFeature', 'ADD'))
    
    #Split the numpy arrays into documents
    examplesByDoc = []
    yListsByDoc = []

    if fileNameVariance != "baseline":
        firstSentenceIndex = 0
        for i in xrange(X.shape[0]):
            if X[i, lastSentenceFeatureIndex] != 0:
                examplesByDoc.append(X[firstSentenceIndex:i + 1,:])
                yListsByDoc.append(np.asarray(y[firstSentenceIndex:(i + 1)]))
            if X[i, firstSentenceFeatureIndex] != 0:
                firstSentenceIndex = i
    else:
        #Split the numpy arrays into documents without added features
        firstSentenceIndex = 0
        for i in xrange(X.shape[0]):
            if firstOrLastSentence[i] == 2:
    		examplesByDoc.append(X[firstSentenceIndex:i + 1,:])
    		yListsByDoc.append(np.asarray(y[firstSentenceIndex:(i + 1)]))
            if firstOrLastSentence[i] == 1:
                firstSentenceIndex = i
    
    print "Done constructing design matrix."
    
    return examplesByDoc, yListsByDoc


def predict(classifier, X):
    return classifier.predict(X)

def train(classifer, X, y):
    classifier.fit(X, y)

def benchmark(classifier, X_train, y_train, trainTuples, testTuples, exampleTuple = ()):
    classifier_descr = str(classifier) 
    classifier_descr_first = str(classifier).split('(')[0]
    def testOnSet(X, y, kBest=-1, returnPred = False):
        #Prediction
        pred = predict(classifier, X)

        if classifier_descr_first == "LinearSVC" or classifier_descr_first == "SGDClassifier":
            #Predict by k-highest score on decision function
            predicted_test_scores= classifier.decision_function(X) 

            if kBest != -1:
                kBest = sum(y.tolist()) #Tell the classifier the expected number of important sentences
                best_ind_array = np.argsort(-predicted_test_scores)[:kBest]
                pred = np.zeros(pred.size, dtype = int)
                for bestIndex in best_ind_array:
                    pred[bestIndex] = 1
        elif classifier_descr_first != "NearestCentroid":
            if kBest != -1:
                kBest = y.tolist().count(1)

                #Naive Bayes uses probabilities instead of decision functions
                predicted_test_scores = classifier.predict_proba(X)
                best_ind_array = np.argsort(-predicted_test_scores[:,1])[:kBest]
                pred = np.zeros(pred.size, dtype = int)
                for bestIndex in best_ind_array:
                    pred[bestIndex] = 1
        
        if returnPred:
            return pred


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

    if exampleTuple:
        examplePred = testOnSet(exampleTuple[0], exampleTuple[1], kBest=1, returnPred = True)

    print('_' * 40)
    print("Testing on Training Set: ")
    accumulatedTrain = np.array([[0,0],[0,0]])

    for X_tr, y_tr, docIndex in trainTuples:
        confusion_matrix = testOnSet(X_tr, y_tr, kBest=1)
        accumulatedTrain = np.add(accumulatedTrain, confusion_matrix)
    
    print("Testing on Test Set: ")	

    accumulatedTest = np.array([[0,0],[0,0]])

    for X_test, y_test, docIndex in testTuples:
        confusion_matrix = testOnSet(X_test, y_test, kBest=1)
        accumulatedTest = np.add(accumulatedTest, confusion_matrix)

    print exampleTuple

    
    if exampleTuple:
        return (classifier_descr, accumulatedTrain, accumulatedTest, examplePred)
    else:
        return (classifier_descr, accumulatedTrain, accumulatedTest)
    

def runTests(examplesByDoc, yListsByDoc):
    print "Running Tests..."
    results = []

    examplePredictions = []
    exampleDocIndex = 0
    exampleTuple = ()
    exampleNeeded = True
    exampleGenerated = False

    #10-fold cross validation on number of documents (2024)
    kf = cross_validation.KFold(len(examplesByDoc), n_folds=10)	

    for train_index, test_index in kf:

        #We pick 90% of documents to train on, 10% to test on each fold 

        #build k-fold partitions by documents. We train by the entire 
        #training set but test performance on a per-document basis
        trainTuples = []
        testTuples = []
        first = True
        for i in test_index:
            #Extract one document to generate an example summary for
            if exampleNeeded and sum(yListsByDoc[i].tolist()) != 0:
                exampleDocIndex = i
                exampleNeeded = False
                exampleTuple = (examplesByDoc[i], yListsByDoc[i], i) 
            testTuples.append((examplesByDoc[i], yListsByDoc[i], i))
		
        for i in train_index:
            trainTuples.append((examplesByDoc[i], yListsByDoc[i], i))
            if first:
                X_train = examplesByDoc[i]
                y_train = yListsByDoc[i]
                first = False
            else:
                X_train = sp.vstack((X_train, examplesByDoc[i]), format='csr')
                y_train = np.concatenate((y_train, yListsByDoc[i]))

        if fileNameVariance != "baseline":
            # Train Liblinear model with L2 regularization
            for penalty in ["l2"]:
                print('=' * 80)
                print("%s regularization" % penalty.upper())
                classifier = LinearSVC(loss='l2', penalty=penalty, dual=True, C = 1, class_weight='auto')
                if exampleGenerated:
                    results.append(benchmark(classifier, X_train, y_train, trainTuples, testTuples))
                else:
                    classifier_descr, accumulatedTrain, accumulatedTest, examplePred = benchmark(classifier, X_train, y_train, trainTuples, testTuples, exampleTuple)
                    examplePredictions = examplePred
                    results.append((classifier_descr, accumulatedTrain, accumulatedTest))
                    exampleGenerated = True

            # Train sparse Naive Bayes classifiers
            print('=' * 80)
            print("Naive Bayes")
            classifier = MultinomialNB(alpha=.01)
            results.append(benchmark(classifier, X_train, y_train, trainTuples, testTuples))
            classifier = BernoulliNB(alpha=.01)
            results.append(benchmark(classifier, X_train, y_train, trainTuples, testTuples))
            
        #Baseline linear hinge loss classifier
        classifier = SGDClassifier(loss='hinge')
        results.append(benchmark(classifier, X_train, y_train, trainTuples, testTuples))

    print('='*80)
    print 'Aggregate'
    aggregate_results = {}
    for tup in results:
        classifier_descr, confusion_matrix_train, confusion_matrix = tup
    	accumulatedTrain = aggregate_results.get(classifier_descr + 'train', np.array([[0,0],[0,0]]))
        accumulatedTest = aggregate_results.get(classifier_descr + 'test', np.array([[0,0],[0,0]]))
        aggregate_results[classifier_descr + 'train'] = np.add(accumulatedTrain, confusion_matrix_train)
        aggregate_results[classifier_descr + 'test'] = np.add(accumulatedTest, confusion_matrix)

    for classifier_descr, confusionMatrix in aggregate_results.iteritems():
        print('-'*40)
        print classifier_descr
        print confusionMatrix
        recall = confusionMatrix[1,1] / float(confusionMatrix[1,1] + confusionMatrix[1,0])
        precision = confusionMatrix[1,1] / float(confusionMatrix[1,1] + confusionMatrix[0,1])
        print "Precision: ", precision, " Recall: ", recall

    return (exampleDocIndex, examplePredictions)



#Controller

#fileNameVariance can be oracle_with_title to run on oracle subset
fileNameVariance = "with_title"

parser = OptionParser()
parser.add_option('-n', action="store", dest="numExamples", type="int", default=2021, help="Number of documents to process. Default:all")
parser.add_option('--nf', action="store_false", dest="format", default=True, help="Don't reformat examples") 
parser.add_option('-s', action="store_true", dest="savepkl", default=False, help="Save formatting to pkl") 
parser.add_option('-o', action="store_true", dest="useOracle", default=False, help="Test on oracle dataset")
parser.print_help()
options, remainder = parser.parse_args()
examplesByDoc, yListsByDoc = loadData()
categories = [
        'not important',
        'important'
        ]
generateSummaryDocIndex, pred = runTests(examplesByDoc, yListsByDoc)

if not options.useOracle:
    unpreprocessedSentences = processExamples.getObjFromPklz('unpreprocessed_sentences.pklz')
else:
    unpreprocessedSentences = processExamples.getObjFromPklz('unpreprocessed_oraclesentences.pklz')

with open('generated_summary.txt', 'a') as f:
    for i, sentence in enumerate(unpreprocessedSentences[generateSummaryDocIndex]):
	if pred[i] == 1:
            f.write(sentence + '\n')
    f.write('\n' + '\n' + '--------------------------------------ORIGINAL---------------------------------------------' + '\n' + '\n')
    for i, sentence in enumerate(unpreprocessedSentences[generateSummaryDocIndex]):
        f.write(sentence + '\n')


