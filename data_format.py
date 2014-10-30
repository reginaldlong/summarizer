import re
# Given list of all sentences and catchphrases, return (x, y) pairs 


stopWords = []
def initStopWords():
    with open("stopwords.txt") as f:
        for line in f:
            stopWords.append(line)

def removeStopWords(sentence):
    ret = ""
    for stopWord in stopWords:
        ret = re.sub(stopWord, "", sentence)
    return ret

# Sentences is a list of lists of sentences
# Catchphrases is a list of lists of catchphrases
# Each element represents the list of sentences/catchphrases
# from one file.
def format(sentences, catchphrases):
    examples = []

    #read stop words from file
    initStopWords()

    # change this later to len(sentences)
    for i in xrange(400):
        currentFileSentences = sentences[i]
        currentFileCatchphrases = catchphrases[i]
        for sentence in currentFileSentences:
            if sentence == None:
                continue
            formattedSentence = removeStopWords(sentence)
            value = 0
            for catchphrase in currentFileCatchphrases:
                formattedCatchphrase = removeStopWords(catchphrase)
                if formattedCatchphrase in formattedSentence:
                    value = 1
            examples.append((formattedSentence, value))
    return examples




