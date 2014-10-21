import retrieve_data
import collections
import util
import data_format

def extractWordFeatures(x):
    vec = collections.Counter()
    for word in x.split():
        vec[word] = vec.get(word, 0) + 1
    return vec


def learnPredictor(trainExamples, testExamples, featureExtractor):
    weights = collections.Counter()
    def loss(w, phi, y):
        return max(1 - util.dotProduct(w, phi) * y, 0)
    eta = 0.1  
    numIters = 30 
    def sgradLoss(w, phi, y):
        if loss(w, phi, y) == 0:
            return collections.Counter()
        for key, value in phi.items():
            phi[key] = -1 * phi[key] * y
        return phi
    def predictor(x):
        if x == None:
            return -1
        return (1 if util.dotProduct(featureExtractor(x), weights) > 0 else -1)

    for iteration in xrange(numIters):
        for input, output in trainExamples:
            if input == None:
                continue
            util.increment(weights, -1 * eta, sgradLoss(weights, 
                featureExtractor(input), output))
        print util.evaluatePredictor(trainExamples, predictor) 
    return weights

sentences = []
catchphrases = []
retrieve_data.parseFiles(sentences, catchphrases)
totalFiles = len(catchphrases)
numLearnFiles = totalFiles / 2
trainExamples = data_format.format(sentences, catchphrases)
w = learnPredictor(trainExamples, None, extractWordFeatures)
print w
