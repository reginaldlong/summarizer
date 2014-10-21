# Given list of all sentences and catchphrases, return (x, y) pairs 


# Sentences is a list of lists of sentences
# Catchphrases is a list of lists of catchphrases
# Each nested list represents the sentences/catchphrases
# from one file.
def format(sentences, catchphrases):
    examples = []
    # change this later to len(sentences)
    for i in xrange(400):
        currentFileSentences = sentences[i]
        currentFileCatchphrases = catchphrases[i]
        for sentence in currentFileSentences:
            value = 0
            for catchphrase in currentFileCatchphrases:
                if sentence != None:
                    if catchphrase in sentence:
                        value = 1
            examples.append((sentence, value))
    return examples

