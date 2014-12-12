import re
from nltk.stem.wordnet import WordNetLemmatizer
# Given list of all sentences and catchphrases, return (x, y) pairs 


stopWords = {}
def initStopWords():
    with open("stopwords.txt") as f:
        for line in f:
            stopWords[line.strip()] = 1

def removeStopWords(sentence):
    wordList = sentence.lower().strip().split(" ")   
    retList = []
    for word in wordList:
       if stopWords.get(word, -1) != 1:
            retList.append(word)
    return retList
    
# Sentences is a list of lists of sentences
# Catchphrases is a list of lists of catchphrases
# Each element represents the list of sentences/catchphrases
# from one file.
def format(sentences, catchphrases, titles):
    examples = []
    lemmatizer = WordNetLemmatizer()
    #read stop words from file
    initStopWords()

    # change this later to len(sentences)
    for i in xrange(len(sentences)):
        currentFileSentences = sentences[i]
        currentFileCatchphrases = catchphrases[i]
        associatedSentences = []
        associatedCatchphrases = []
        for sentence in currentFileSentences:
            if sentence == None:
                continue
            formattedSentenceList = removeStopWords(sentence)
            formattedSentence = " ".join([lemmatizer.lemmatize(kw) for kw in \
                    formattedSentenceList])
            associatedSentences.append(formattedSentence)
        for catchphrase in currentFileCatchphrases:
            formattedCatchphraseList = removeStopWords(catchphrase)
            formattedCatchphrase = " ".join([lemmatizer.lemmatize(kw) for kw \
                in formattedCatchphraseList])
            associatedCatchphrases.append(formattedCatchphrase)
        examples.append((associatedSentences, associatedCatchphrases, titles[i].lower().strip()))
    return examples

# Sentences is a list of lists of sentences
# Catchphrases is a list of lists of catchphrases
# Each element represents the list of sentences/catchphrases
# from one file.
def format_Old(sentences, catchphrases):
    examples = []
    
    for i in xrange(len(sentences)):
        currentFileSentences = sentences[i]
        currentFileCatchphrases = catchphrases[i]
        associatedSentences = []
        associatedCatchphrases = []
        for sentence in currentFileSentences:
            if sentence == None:
                continue
            associatedSentences.append(sentence.lower().strip())
        for catchphrase in currentFileCatchphrases:
            associatedCatchphrases.append(catchphrase.lower().strip())
        examples.append((associatedSentences, associatedCatchphrases))
    return examples




