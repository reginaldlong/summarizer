import retrieve_data
import collections
import util
import data_format
import pickle
import gzip

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

def processFiles():
    sentences = []
    catchphrases = []
    titles = []
    filesInExamples = retrieve_data.parseFiles(sentences, catchphrases, titles)
    
    numSentences = 0
    for s in sentences:
        numSentences += len(s)
    print numSentences

    #examples = data_format.format(sentences, catchphrases, titles)
    old_examples = data_format.format_Old(sentences, catchphrases)


    writeToPklz('unpreprocessed_sentences.pklz', sentences)
    writeToPklz('unpreprocessed_catchphrases.pklz', catchphrases)
    #writeToPklz('oracle_examples_with_title.pklz', examples)
    #writeToPklz('2021filesInExamples_oracle.pklz', filesInExamples)
    writeToPklz('unpreprocessed_examples.pklz', old_examples)

#processFiles()
