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
    words = okt.pos(review[1]) #review를 토크나이즈
    sentence = []
    for word in words :
        tokenized_review.append(word[0]) #토크나이즈 된 값 (word, tag)에서 word만 list에 저장
        sentence.append(word[0]) #토크나이즈 된 값 list에 저장 ==> string으로 변환 예정
    sentence = ' '.join(sentence) # 각 단어간 공백을 추가하여 string으로 변환 ==> TF-IDF 사용
    review[1] = tokenized_review
    tokenized_reviews.append(tokenized_review) #tokenize 된 값을 저장 ==> window에 사용예정
    reviews.append(sentence) #string 모음

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

#build vocab
vocab = list()
word_dict = {}
num_doc = 0

#add documents in vocab

num_doc = len(train_data)
#add words in vocab
# 단어가 있으면 vocab에 넣고 있으면 빈도수를 올린다.
for review in train_data :
    for idx in range(review[1]) :
        if review[1][idx] not in vocab :
            vocab.append(review[1][idx])
        if review[1][idx] not in word_dict :
            word_dict[review[1][idx]] = 1
        else :
            word_dict[review[1][idx]] = word_dict[review[1][idx]] + 1

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




# 빈도수 1 이하인 단어 제거
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

"""
data = read_data('data/new_vocab.pkl')
vocab = data['new_vocab']
train_data = data['train_data']
num_doc = len(train_data)
"""
#build A
A = np.eye(len(vocab)+num_doc)

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(reviews)
X = X.toarray()
#Y = tfidf.get_feature_names()

windows = make_window(tokenized_reviews, 20) #전체 리뷰를 바탕으로 window_size 만큼의 윈도우 생성
num_windows = len(windows)

word_freq = count_word_freq(vocab, windows) #단어가 포함된 window 계산
print("===========================================")
print("Counting Word Frequency Complete!")
print("===========================================")
f = open("data/till_word_freq.pkl", 'wb')
data['word_freq'] = word_freq
pickle.dump(data, f)
f.close()
pair_freq = count_pair_freq(windows) #단어 두개씩 포함된 window 계산
print("===========================================")
print("Counting pair Frequency Complete!")
print("===========================================")

f = open("data/till_word_pair.pkl", 'wb')
data['pair_freq'] = word_freq
pickle.dump(data, f)
f.close()
# A의 구성 :
#       word    doc
# word
# doc

for i in range(len(vocab)+num_doc) :
    for j in range(i+1, len(vocab)+num_doc) :
        if i < len(vocab) and j < len(vocab) :
            # i , j is word A[i][j] is PMI
            pi = 1.0 * (word_freq[vocab[i]] / num_windows)
            pj = 1.0 * (word_freq[vocab[j]] / num_windows)
            pij = 1.0 * (pair_freq[(vocab[i], vocab[j])] / num_windows)
            if pij > 0 :
                pmi = log(pij/(pi*pj))
                if pmi >= 0 :
                    A[i][j] = pmi
                    A[j][i] = pmi
        elif i >= len(vocab) and j < len(vocab) and vocab[j] in tfidf.vocabulary_ :
            # i is document and j is word A[i][j] = TF-IDF
            idx = tfidf.vocabulary_.get(vocab[j])
            A[i][j] = X[i-num_doc][idx]
            A[j][i] = X[i-num_doc][idx]
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