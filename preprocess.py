import numpy as np
import nltk
from nltk.corpus import reuters
import string
import re

def getTestTrain(fileds):
    testids = []
    trainids = []
    for id in fileids:
        tokens = id.split('/')
        if tokens[0] == 'training':
            trainids.append(id)
        else:
            if tokens[0] == 'test':
                testids.append(id)
    return testids, trainids

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

# remove numbers. they can be like 100 1090,200 2.123 etc
# strategy is to remove punctuation and then check if its an integer
def isNumber(word):
    word_no_num = re.sub(r'[^\w\s]','',word)
    if RepresentsInt(word_no_num):
        return True
    else:
        return False

#tokenizes raw strings
def getTokenized(ids):
    exclude = set(string.punctuation)
    words_list = []
    for id in ids:
        raw = reuters.raw(id)
        words = nltk.word_tokenize(raw)
        words_nopunc_nonum = []
        for word in words:
            if word in exclude: # if punctuation
                continue
            else:
                word = word.lower()
                if(isNumber(word)): # if number
                    continue
                words_nopunc_nonum.append(word)
        words_list.append(words_nopunc_nonum)
    return words_list

def getVocabulary(tokenized_corpus):
    vocabulary = []
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)
    word2id = {w: idx for (idx, w) in enumerate(vocabulary)}
    id2word = {idx: w for (idx, w) in enumerate(vocabulary)}
    return vocabulary, word2id, id2word


def getVocabulary(tokenized_corpus):
    vocabulary = {}
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                vocabulary[token] = 1
    word2id = {w: idx for (idx, w) in enumerate(vocabulary)}
    id2word = {idx: w for (idx, w) in enumerate(vocabulary)}
    return vocabulary, word2id, id2word

def generatePairs(sentences_tokenized, word2id, id2word):
    window_size = 2
    idx_pairs = []
    # for each sentence
    for sentence in sentences_tokenized:
        indices = [word2id[word] for word in sentence]
        # for each word, threated as center word
        for center_word_pos in range(len(indices)):
            # for each window position
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w
                # make soure not jump out sentence
                if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                    continue
                context_word_idx = indices[context_word_pos]
                idx_pairs.append((indices[center_word_pos], context_word_idx))
    return idx_pairs

def writeToFile(vocabulary, word2id, id2word, pairs)
    f1 = open('data/vocab.txt', 'w')
    f2 = open('data/word2id.txt', 'w')
    f3 = open('data/id2word.txt', 'w')

    for key, value in vocabulary.items():
        f1.write(key + '\n')
    for key, value in word2id.items():
        f2.write(key + '\t' + str(value) + '\n')
    for key, value in id2word.items():
        f3.write(str(key) + '\t' + value + '\n')

    f4 = open("data/pairs.txt", "w")
    for pair in pairs:
        f4.write(str(pair[0]) + '\t' + str(pair[1]) + '\n')
    f4.close()
    f3.close()
    f2.close()
    f1.close()


print('Getting ids')
fileids = reuters.fileids()
testids, trainids = getTestTrain(fileids)
print('Tokenizing')
sentences_tokenized = getTokenized(trainids)
print('Generating vocab')
vocabulary, word2id, id2word = getVocabulary(sentences_tokenized)
pairs = generatePairs(sentences_tokenized, word2id, id2word)


print('Writing data to files')
# writeToFile(vocabulary, word2id, id2word, pairs)







