README.txt
------------------------------------------------------------------------------
Reginald Long
Michael Xie
Helen Jiang

[reglong, sxie, helennn]

------------------------------------------------------------------------------


Listed below are the pertinent code files:
-baseline.py
-data_format.py
-processExamples.py
-retrieve_data.py
-util.py
-stopwords.txt
-mlclassifiers_test.py
-generatedSummary.txt

processExamples.py
--------------------------------------------------
This file controls the parsing of data files and running preprocessing of the documents into (sentences, catchphrase, title) tuples, the actual
jobs which are done in retrieve_data and data_format. 
After parsing and preprocessing, this script saves the sentences, catchphrases, and example pairs in .pklz files for later use.
processExamples also provides utility functions for getting and saving compressed pickle files using gzip. 

retrieve_data.py
--------------------------------------------------
This file contains the routines for parsing the XML data files using the lxml ElementTree library. 
The parseFiles() function parses all files in a given directory and stores a list of sentences, catchphrases, and titles for each document.
This file also contains the cleanFiles() function, which is used to correct XML formatting errors found in the original dataset. 

data_format.py
--------------------------------------------------
This file controls preprocessing of the data. Specifically, its format() function converts each sentence and catchphase to lowercase, lemmatizes each word, 
removes stopwords (defined by the words in stopwords.txt). It outputs examples as (sentences, catchphrases, title) tuples.

baseline.py
--------------------------------------------------
This file contains an implementation of a hinge loss linear classifier that learns using stochastic gradient descent. 

util.py
--------------------------------------------------
util contains helper functions for the baseline hinge loss linear classifier, such as dotProduct(x,y).

mlclassifiers_test.py
--------------------------------------------------
This file contains the feature extraction/example formatting code as well as the testing unit for classifiers supplied by the sklearn library. 
The execution of this script takes several options, including -s to save the design matrix, label vector, and vectorizer in .pklz files, --nf to 
use a design matrix/label vector/vectorizer in existing pklz files, -o to use a design matrix with old features(less document context), -n to specify the 
number of documents in which to include in the dataset. 

The behavior of mlclassifiers_test also differs given different values for fileNameVariance and examplesFileName. These tell mlclassifiers_test where to load
the design matrix/label vector from, where to save to, where to load the sentences and catchphrases from. This is important for running tests on differently processed
data. 

Included also in the feature extraction is determining the labels of each sentence - we use a similarity ranking of sentences to catchphrases. We choose the 
most similar sentence to each catchphrase as important sentences, so that there is one important sentence per catchphrase. 

The test unit takes formatted examples and uses 10-fold cross validation, splitting the dataset by documents. It then aggregates the confusion matrices from each 
classifier throughout all tests and outputs the aggregate results. 

The test unit also chooses a document in which to output an example summary for, which it saves in generatedSummary.txt. 


Evaluation report on the oracle subset
---------------------------------------------------------
Oracle: 5 human summarizers for each document 
[[108    51]
 [   61   50]]
Precision: 0.4950   Recall: 0.4505 

Aggregate 10-fold cross validation on 10 documents used in the oracle:

LinearSVC(C=1, class_weight='auto', dual=True, fit_intercept=True,
     intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',
     random_state=None, tol=0.0001, verbose=0)train
[[18886   410]
 [  410   130]]
Precision:  0.240740740741  Recall:  0.240740740741
----------------------------------------
LinearSVC(C=1, class_weight='auto', dual=True, fit_intercept=True,
     intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',
     random_state=None, tol=0.0001, verbose=0)test
[[2093   51]
 [  51    9]]
Precision:  0.15  Recall:  0.15
----------------------------------------

MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)train
[[19283    13]
 [   13   527]]
Precision:  0.975925925926  Recall:  0.975925925926
----------------------------------------
MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)test
[[2092   52]
 [  52    8]]
Precision:  0.133333333333  Recall:  0.133333333333
----------------------------------------

BernoulliNB(alpha=0.01, binarize=0.0, class_prior=None, fit_prior=True)train
[[19276    20]
 [   20   520]]
Precision:  0.962962962963  Recall:  0.962962962963
----------------------------------------
BernoulliNB(alpha=0.01, binarize=0.0, class_prior=None, fit_prior=True)test
[[2097   47]
 [  47   13]]
Precision:  0.216666666667  Recall:  0.216666666667
----------------------------------------
BaselineSGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
       fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
       loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5,
       random_state=None, shuffle=False, verbose=0, warm_start=False)train
[[18821   475]
 [  475    65]]
Precision:  0.12037037037  Recall:  0.12037037037
----------------------------------------
BaselineSGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
       fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
       loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5,
       random_state=None, shuffle=False, verbose=0, warm_start=False)test
[[2089   55]
 [  55    5]]
Precision:  0.0833333333333  Recall:  0.0833333333333
----------------------------------------

