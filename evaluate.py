from scipy.stats import spearmanr
import numpy as np
import nltk
from nltk.corpus import reuters
import string
import re
from random import randint
import math
import argparse, sys
import time
import argparse, sys


parser=argparse.ArgumentParser()

parser.add_argument('--wfile', help='word embedding file')
parser.add_argument('--cfile', help='context embedding file')

args=parser.parse_args()
wfile = 'data/W_NEG_20_DIM_300_EPOCHS_25_nonum_full_win3_epoch75.npy.npz'
cfile = 'data/C_NEG_20_DIM_300_EPOCHS_25_nonum_full_win3_epoch75.npy.npz'
cmd_vars = vars(args)
if(cmd_vars['wfile']):
    wfile = cmd_vars['wfile']
if(cmd_vars['cfile']):
    cfile = cmd_vars['cfile']


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

def cosineDistance(x, y):
    s = np.dot(x,y)
    s = s/((np.dot(x,x))**0.5)
    s = s/((np.dot(y,y))**0.5)
    return s

def cosineDistanceString(s1, s2, W, word2id):
    x = W[word2id[s1]]
    y = W[word2id[s2]]
    return cosineDistance(x, y)


def evalSimlex(W, vocab, word2id, id2word):
    f = open("SimLex-999.txt", "r")
    scores = []
    i = 0
    for line in f:
        i+= 1
        if(i == 1):
            continue
        line = line.strip()
        tokens = line.split('\t')
        sample = {}
        sample['w1'] = tokens[0]
        sample['w2'] = tokens[1]
        sample['score'] = float(tokens[3])
        if sample['w1'] in vocab and sample['w2'] in vocab:
            scores.append(sample)
    predicted_scores = []
    simlex_score = []
    for sample in scores:
        w1 = sample['w1']
        w2 = sample['w2']
        predicted_scores.append(cosineDistanceString(w1, w2, W, word2id))
        simlex_score.append(sample['score'])
    corr, p_value = spearmanr(simlex_score, predicted_scores)
    return corr


vocab, word2id, id2word = readAll()
npz = np.load(wfile)
W = npz['arr_0']
W = W.T
npz = np.load(cfile)
C = npz['arr_0']

print(evalSimlex(W, vocab, word2id, id2word))
