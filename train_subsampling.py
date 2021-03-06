import numpy as np
import nltk
from nltk.corpus import reuters
import string
import re
from random import randint
import math


POWER_FOR_NEGATIVE_SAMPLING = 3.0/4.0

def readPairs(fileName):
    f1 = open(fileName, "r")
    pairs = []
    for line in f1:
        pair = line.strip().split('\t')
        for i in range(len(pair)):
            pair[i] = int(pair[i])
        pairs.append(pair)
    f1.close()
    return pairs

def readVocabulary(fileName):
    f1 = open(fileName, "r")
    vocab = {}
    for line in f1:
        v = line.strip().split('\t')
        vocab[v[0]] = int(v[1])
    f1.close()
    return vocab

def readWord2id(fileName):
    f1 = open(fileName, "r")
    word2id = {}
    for line in f1:
        v = line.strip().split('\t')
        word2id[v[0]] = int(v[1])
    f1.close()
    return word2id

def readId2word(fileName):
    f1 = open(fileName, "r")
    id2word = {}
    for line in f1:
        v = line.strip().split('\t')
        id2word[int(v[0])] = v[1]
    f1.close()
    return id2word

def readAll():
    vocab = readVocabulary('data/vocab.txt')
    word2id = readWord2id('data/word2id.txt')
    id2word = readId2word('data/id2word.txt')
    return vocab, word2id, id2word

def getArrForNegativeSampling(vocab):
    idsForNegativeSampling = []
    for word, count in vocab.items():
        newCount = int(float(count)**(POWER_FOR_NEGATIVE_SAMPLING))
        for i in range(newCount):
            idsForNegativeSampling.append(word2id[word])
    return idsForNegativeSampling

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


print('Reading pairs')
pairs = readPairs('data/pairs_subsampled.txt')
print('Reading vocab')
vocab, word2id, id2word = readAll()
print('Creating array for negative sampling')
idsForNegativeSampling = getArrForNegativeSampling(vocab)

print(len(idsForNegativeSampling), len(vocab), len(pairs))


# logic for training:
# create random matrics W and C
# W: centre word embedding: d x vocab_size
# C: context word embedding: vocab_size x d
vocab_size = len(vocab)
EMBEDDING_DIMENSION = 300
LR = 0.01
NUM_NEGATIVE_SAMPLES = 5
W = np.random.rand(EMBEDDING_DIMENSION, vocab_size)
C = np.random.rand(vocab_size, EMBEDDING_DIMENSION)
print('Examples in 1 epoch', len(pairs))
for epoch in range(100):
    numDone = 0
    TOTAL_OBJECTIVE = 0
    for pair in pairs:
        a = int(pair[0]) # centre word id
        b = int(pair[1]) # context word id

        n = [] # to store negative context word ids
        for i in range(NUM_NEGATIVE_SAMPLES):
            x = randint(0, len(idsForNegativeSampling)-1)
            n.append(idsForNegativeSampling[x])
        
        # n now stores ids for negative samples
        wa = W.T[a]
        # get gradient for wa
        sigCbwa = sigmoid(np.dot(C[b], wa))
        gradwa = (1.0 - sigCbwa)*C[b]
        sigminusCniwa = {}
        for id in n:
            sigminusCniwa[id] = sigmoid(np.dot(-C[id], wa))
            gradwa += (sigminusCniwa[id] - 1.0)*C[id]
            
        # update context embedding for positive sample:
        C[b] += LR*(1.0 - sigCbwa)*wa
        
        #update context embedding for negative samples:
        for id in n:
            C[id] += LR*(sigminusCniwa[id] - 1.0)*wa
        #update wa
        W.T[a] += LR*gradwa
        CUR_OBJECTIVE = math.log(sigCbwa)
        for id in n:
            CUR_OBJECTIVE += math.log(sigminusCniwa[id])
        if(numDone%50000 == 0):
            print('Objective', CUR_OBJECTIVE)
        numDone += 1
        TOTAL_OBJECTIVE += CUR_OBJECTIVE
    print('Epoch done', epoch)
    print('Averaged Total objective', TOTAL_OBJECTIVE/len(pairs))