import numpy as np
import nltk
from nltk.corpus import reuters
import string
import re

def readPairs(fileName):
    f1 = open(fileName, "r")
    pairs = []
    for line in f1:
        pair = line.strip().split('\t')
        pairs.append(pair)