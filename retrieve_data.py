import xml.etree.cElementTree as ET 
import re, mmap
from lxml import etree

from os import listdir
from os.path import isfile, join


dirpath = "corpus/fulltext/"	
files = [f for f in listdir(dirpath) if isfile(join(dirpath,f))]



def cleanFiles():
    for file in files:
        with open(dirpath + file, 'r+') as f:
            data = mmap.mmap(f.fileno(), 0)
            newData = re.sub('".*?=.*?"', "", data)
            newData = re.sub('&', "", newData)
            f.write(newData)


#cleanFiles()

def parseFiles(allSentences, allCatchphrases):
    for file in files:
        parser = etree.XMLParser(recover = True)
        tree   = etree.parse(dirpath + file, parser)	
        root = tree.getroot()
        catchphrases = []
        sentences = []
        for child in root:
            if child.tag == "catchphrases":
                catchphrases = [catchphrase.text for catchphrase in child]
            elif child.tag == "sentences":
                sentences = [sentence.text for sentence in child]
        if sentences and sentences[0] != None:
            allSentences.append(sentences)
        if catchphrases and catchphrases[0] != None:
            allCatchphrases.append(catchphrases)
        #Do stuff	
sentences = []
catchphrases = []
parseFiles(sentences, catchphrases)

