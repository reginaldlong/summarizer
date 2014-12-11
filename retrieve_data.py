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

def parseFiles(allSentences, allCatchphrases, allTitles):
    filesInExamples = []
    for file in files:
        parser = etree.XMLParser(recover = True)
        tree   = etree.parse(dirpath + file, parser)	
        root = tree.getroot()
        catchphrases = []
        sentences = []
        title = None
        for child in root:
            if child.tag == "catchphrases":
                catchphrases = [catchphrase.text for catchphrase in child]
            elif child.tag == "sentences":
                sentences = [sentence.text for sentence in child]
            elif child.tag == "name":
                try:
                    title = etree.tostring(child, encoding='unicode', method='text')
                except:
                    pass
        if sentences and catchphrases and title:
            allSentences.append(sentences)
            allCatchphrases.append(catchphrases)
            allTitles.append(title)
            filesInExamples.append(file)

    return filesInExamples
            

