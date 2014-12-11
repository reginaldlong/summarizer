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
    print len(titles), len(filesInExamples)
    examples = data_format.format(sentences, catchphrases, titles)
    #old_examples = data_format.format_Old(sentences, catchphrases)


    #writeToPklz('sentences.pklz', sentences)
    #writeToPklz('catchphrases.pklz', catchphrases)
    writeToPklz('new_examples_with_title.pklz', examples)
    writeToPklz('filesInExamples,pklz', filesInExamples)
    #writeToPklz('unpreprocessed_examples.pklz', old_examples)


processFiles()
