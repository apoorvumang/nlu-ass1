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
parser.add_argument('--word', help='word')

args=parser.parse_args()
wordToFindClosest = 'would'
wfile = 'data/W_NEG_20_DIM_300_EPOCHS_25_nonum_full_win3_epoch75.npy.npz'
cfile = 'data/C_NEG_20_DIM_300_EPOCHS_25_nonum_full_win3_epoch75.npy.npz'
cmd_vars = vars(args)
if(cmd_vars['wfile']):
    wfile = cmd_vars['wfile']
if(cmd_vars['cfile']):
    cfile = cmd_vars['cfile']
if(cmd_vars['word']):
    wordToFindClosest = cmd_vars['word']



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


def findTopK(word_to_eval, W, word2id, id2word):
    dict = {}
    for i in range(len(W)):
        word = id2word[i]
        dist = cosineDistance(W[word2id[word_to_eval]], W[i])
        dict[word] = dist
    sorted_by_value = sorted(dict.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_by_value[1:20]


vocab, word2id, id2word = readAll()
npz = np.load(wfile)
W = npz['arr_0']
W = W.T
npz = np.load(cfile)
C = npz['arr_0']

print(findTopK(wordToFindClosest, C, word2id, id2word))
