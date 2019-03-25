from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
from math import log
from utils import make_window, count_word_freq, count_pair_freq
from konlpy.tag import Okt



#load dataset
# id review label
def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data


train_data = read_data('data/ratings_train.txt')

print("===========================================")
print("Load Data Complete!")
print("===========================================")

train_data = train_data[:10000] #samples
#tokenize
okt = Okt()
reviews = []
tokenized_reviews = []
for review in train_data :
    tokenized_review = []
    words = okt.pos(review[1])
    sentence = []
    for word in words :
        tokenized_review.append(word[0])
        sentence.append(word[0])
    sentence = ' '.join(sentence)
    review[1] = tokenized_review
    tokenized_reviews.append(tokenized_review)
    reviews.append(sentence)

"""
train_data = (id, tokenized_review, label)
sentence = tokenized string
reviews = list of sentences
tokenized_reviews = [tokenized_review]

"""

print("===========================================")
print("Data Tokenize Complete!")
print("===========================================")

data = {}
data['train_data'] = train_data

f = open("data/token.pkl", 'wb')
pickle.dump(data, f)
f.close()

train_data = data['train_data']

#build vocab
vocab = list()
word_dict = {}
num_doc = 0

#add documents in vocab
"""
for review in train_data :
    if review[0] not in vocab :
        vocab.append(review[0])
"""
num_doc = len(train_data)
#add words in vocab
for review in train_data :
    for word in review[1] :
        if word not in vocab :
            vocab.append(word)
        if word not in word_dict :
            word_dict[word] = 1
        else :
            word_dict[word] = word_dict[word] + 1

print("===========================================")
print("Build Vocabulary Complete!")
print("Number of documents : ", num_doc)
print("Number of words : ", len(vocab))
print("Vocab Size : ", len(vocab))
print("===========================================")


data['vocab'] = vocab
data['word_dict'] = word_dict

f = open("data/vocab.pkl", 'wb')
pickle.dump(data, f)
f.close()

new_vocab = []
for i in range(len(vocab)) :
    if i < num_doc :
        new_vocab.append(vocab[i])
    else :
        if(word_dict[vocab[i]] > 1) :
            new_vocab.append(vocab[i])

print("Size of New Vocab : ", len(new_vocab))
data['new_vocab'] = new_vocab

f = open("data/new_vocab.pkl", 'wb')
pickle.dump(data, f)
f.close()

#build A
A = np.zeros((len(vocab)+num_doc, len(vocab)+num_doc))

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(reviews)
X = X.toarray()
Y = tfidf.get_feature_names()

windows = make_window(tokenized_reviews, 20)
num_windows = len(windows)

word_freq = count_word_freq(vocab, windows)
print("===========================================")
print("Counting Word Frequency Complete!")
print("===========================================")
pair_freq = count_pair_freq(vocab, windows)
print("===========================================")
print("Counting pair Frequency Complete!")
print("===========================================")



for i in range(len(vocab)+num_doc) :
    for j in range(i, len(vocab)+num_doc) :
        if i == j :
            A[i][j] = 1
        elif i < len(vocab) and j < len(vocab) :
            # i , j is word A[i][j] is PMI
            pi = 1.0 * (word_freq[vocab[i]] / num_windows)
            pj = 1.0 * (word_freq[vocab[j]] / num_windows)
            pij = 1.0 * (pair_freq[(vocab[i], vocab[j])] / num_windows)
            if pij > 0 :
                pmi = log(pij/(pi*pj))
                if pmi >= 0 :
                    A[i][j] = pmi
        elif i >= len(vocab) and j < len(vocab) and vocab[j] in tfidf.vocabulary_ :
            # i is document and j is word A[i][j] = TF-IDF
            idx = tfidf.vocabulary_.get(vocab[j])
            A[i][j] = X[i-num_doc][idx]
    if i%20==0 :
        print(i, "th row finished")

print("===========================================")
print("Build A Complete!")
print("A shape : ", A.shape)
print("===========================================")

X = np.eye(len(vocab)+num_doc)

data = {}
data['train_data'] = train_data
data['A'] = A
data['vocab'] = vocab
data['X'] = X

f = open("data/graph.pkl", 'wb')
pickle.dump(data, f)
f.close()