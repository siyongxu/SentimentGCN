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

def count_pair(windows, i, j) :
    count = 0
    for window in windows :
        if i in window and j in window :
            count += 1

    return count

def count_word_freq(vocab, windows) :
    word_freq = {}
    for word in vocab :
        if word not in word_freq :
            word_freq[word] = count_word(windows, word)

    return word_freq

def count_pair_freq(vocab, windows) :
    pair_freq = {}
    for i in range(len(vocab)) :
        for j in range(i+1, len(vocab)) :
            if (vocab[i], vocab[j]) not in pair_freq :
                pair_freq[(vocab[i], vocab[j])] = count_pair(windows, vocab[i], vocab[j])
    return pair_freq