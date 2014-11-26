import re
from nltk.stem import PorterStemmer
# Given list of all sentences and catchphrases, return (x, y) pairs 


stopWords = {}
def initStopWords():
    with open("stopwords.txt") as f:
        for line in f:
            stopWords[line.strip()] = 1

def removeStopWords(sentence):
    wordList = sentence.lower().split(" ")   
    retList = []
    for word in wordList:
       if stopWords.get(word, -1) != 1:
            retList.append(word)
    return retList
    
# Sentences is a list of lists of sentences
# Catchphrases is a list of lists of catchphrases
# Each element represents the list of sentences/catchphrases
# from one file.
def format(sentences, catchphrases):
    examples = []
    stemmer = PorterStemmer()
    #read stop words from file
    initStopWords()

    # change this later to len(sentences)
    for i in xrange(len(sentences)):
        currentFileSentences = sentences[i]
        currentFileCatchphrases = catchphrases[i]
        for sentence in currentFileSentences:
            if sentence == None:
                continue
            formattedSentenceList = removeStopWords(sentence)
            formattedSentence = " ".join([stemmer.stem(kw) for kw in \
                    formattedSentenceList])
            value = 0
            for catchphrase in currentFileCatchphrases:
                formattedCatchphraseList = removeStopWords(catchphrase)
                formattedCatchphrase = " ".join([stemmer.stem(kw) for kw \
                        in formattedCatchphraseList])
                if formattedCatchphrase in formattedSentence:
                    value = 1
            examples.append((formattedSentence, value))
    return examples




