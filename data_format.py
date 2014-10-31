import re
from nltk.stem import PorterStemmer
# Given list of all sentences and catchphrases, return (x, y) pairs 


stopWords = []
def initStopWords():
    del stopWords[:]
    with open("stopwords.txt") as f:
        for line in f:
            stopWords.append(line[0:len(line) - 1])

def removeStopWords(sentence):
    sentence = sentence.lower()
    ret = ""
    for stopWord in stopWords:
        if ret == "":
            ret = re.sub("(?<!\S)" + stopWord + " ", "", sentence)
        else: 
            ret = re.sub("(?<!\S)" + stopWord + " ", "", ret)
    return ret
    
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
    for i in xrange(1):
        currentFileSentences = sentences[i]
        currentFileCatchphrases = catchphrases[i]
        for sentence in currentFileSentences:
            if sentence == None:
                continue
            formattedSentence = removeStopWords(sentence)
            formattedSentence = " ".join([stemmer.stem(kw) for kw in \
                    formattedSentence.split(" ")])
            value = 0
            for catchphrase in currentFileCatchphrases:
                formattedCatchphrase = removeStopWords(catchphrase)
                formattedCatchphrase = " ".join([stemmer.stem(kw) for kw \
                        in formattedCatchphrase.split(" ")])
                if formattedCatchphrase in formattedSentence:
                    value = 1
            examples.append((formattedSentence, value))
    return examples




