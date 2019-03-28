import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine

def read_data(filename):
    with open(filename, 'r', encoding='UTF8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data


word_embeddings_dim = 300
word_vector_map = {}

# shulffing
doc_name_list = []
doc_train_list = []
doc_test_list = []
dataset = 'ratings'
# 문서 이름 업로드
# 문서 번호로 대체 가능?
"""
for line in lines:
    doc_name_list.append(line.strip())
    temp = line.split("\t")
    if temp[1].find('test') != -1:
        doc_test_list.append(line.strip())
    elif temp[1].find('train') != -1:
        doc_train_list.append(line.strip())
f.close()
"""

#read data and set number of samples
train_data = read_data('data/ratings_train_tokenized.txt')
#train_data = train_data[:10000]
test_data = read_data('data/ratings_test_tokenized.txt')
#test_data = test_data[:2000]
doc_content_list = []
doc_label_list = []
for review in train_data:
	doc_content_list.append(review[1])
	doc_name_list.append(review[0])
	doc_label_list.append(review[2])
for review in test_data:
	doc_content_list.append(review[1])
	doc_name_list.append(review[0])
	doc_label_list.append(review[2])
# print(doc_content_list)

train_ids = []
for review in train_data:
	train_id = doc_name_list.index(review[0])
	train_ids.append(train_id)
print(train_ids)
random.shuffle(train_ids)
# partial labeled data
#train_ids = train_ids[:int(0.2 * len(train_ids))]

train_ids_str = '\n'.join(str(index) for index in train_ids)
f = open('data/' + dataset + '.train.index', 'w')
f.write(train_ids_str)
f.close()

test_ids = []
for review in test_data:
    test_id = doc_name_list.index(review[0])
    test_ids.append(test_id)
print(test_ids)
random.shuffle(test_ids)

test_ids_str = '\n'.join(str(index) for index in test_ids)
f = open('data/' + dataset + '.test.index', 'w')
f.write(test_ids_str)
f.close()

ids = train_ids + test_ids
print(ids)
print(len(ids))

shuffle_doc_name_list = []
shuffle_doc_words_list = []
shuffle_doc_label_list = []
for id in ids:
	shuffle_doc_name_list.append(doc_name_list[int(id)])
	shuffle_doc_label_list.append(doc_label_list[int(id)])
	shuffle_doc_words_list.append(doc_content_list[int(id)])
shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

f = open('data/' + dataset + '_shuffle.txt', 'w', encoding='utf8')
f.write(shuffle_doc_name_str)
f.close()

f = open('data/corpus/' + dataset + '_shuffle.txt', 'w', encoding='utf8')
f.write(shuffle_doc_words_str)
f.close()

# build vocab
word_freq = {}
word_set = set()
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)

word_doc_list = {}

for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i) #단어 마다 자기가 등장한 문서의 번호를 리스트로 관리함
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

vocab_str = '\n'.join(vocab)

f = open('data/corpus/' + dataset + '_vocab.txt', 'w', encoding='utf8')
f.write(vocab_str)
f.close()


# label list
label_list = list()
for doc_meta in shuffle_doc_label_list:
	label_list.append(doc_meta)

label_list_str = '\n'.join(label_list)
f = open('data/corpus/' + dataset + '_labels.txt', 'w', encoding='utf8')
f.write(label_list_str)
f.close()

# x: feature vectors of training docs, no initial features
# slect 90% training set
train_size = len(train_ids)
test_size = len(test_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size  # - int(0.5 * train_size)
# different training rates

real_train_doc_names = shuffle_doc_name_list[:real_train_size]
real_train_doc_names_str = '\n'.join(real_train_doc_names)

f = open('data/' + dataset + '.real_train.name', 'w', encoding='utf8')
f.write(real_train_doc_names_str)
f.close()


'''
Doc word heterogeneous graph
'''

# word co-occurence with context windows
window_size = 20
windows = []

for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)
            # print(window)


word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

word_pair_count = {}
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1

row = []
col = []
weight = []

# pmi as weights

num_window = len(windows)

for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)

# word vector cosine similarity as weights

'''
for i in range(vocab_size):
    for j in range(vocab_size):
        if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
            vector_i = np.array(word_vector_map[vocab[i]])
            vector_j = np.array(word_vector_map[vocab[j]])
            similarity = 1.0 - cosine(vector_i, vector_j)
            if similarity > 0.9:
                print(vocab[i], vocab[j], similarity)
                row.append(train_size + i)
                col.append(train_size + j)
                weight.append(similarity)
'''
# doc word frequency
doc_word_freq = {}

for doc_id in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[doc_id]
    words = doc_words.split()
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_word_set = set()
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        col.append(train_size + j)
        idf = log(1.0 * len(shuffle_doc_words_list) /
                  word_doc_freq[vocab[j]])
        weight.append(freq * idf)
        doc_word_set.add(word)

node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))

# dump objects

f = open("data/ind.{}.adj".format(dataset), 'wb')
pkl.dump(adj, f)
f.close()
