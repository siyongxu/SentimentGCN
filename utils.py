from itertools import combinations
import torch
import numpy as np
import scipy.sparse as sp

def make_window(sentences, window_size) :
    # In this case, sentences = train_data[][1]
    windows = []
    for sentence in sentences :
        sentence_length = len(sentence)
        if sentence_length <= window_size :
            windows.append(sentence)
        else :
            for j in range(sentence_length - window_size + 1) :
                window = sentence[j:j+window_size]
                windows.append(window)
    return windows

def count_word(windows, word) :
    count = 0
    for window in windows :
        if word in window :
            count += 1

    return count

def count_word_freq(vocab, windows) :
    word_freq = {}
    for word in vocab :
        if word not in word_freq :
            word_freq[word] = count_word(windows, word)

    return word_freq

def count_pair_freq(windows) :
    pair_freq = dict()
    for i, window in enumerate(windows) :
        combination = list(combinations(window, 2))
        for comb in combination :
            if (comb[0], comb[1]) in pair_freq :
                pair_freq[(comb[0], comb[1])] += 1
            elif (comb[1], comb[0]) in pair_freq :
                pair_freq[(comb[1], comb[0])] += 1
            else :
                pair_freq[(comb[0], comb[1])] = 1
    return pair_freq

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx