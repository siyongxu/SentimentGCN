from itertools import combinations

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