# SentimentGCN
Sentiment classification based on TextGCN

### build_graph
1. data preprocessing
    - train_data = (id, review, label)
2. tokenize
    - train_data = (id, tokenized_review, label)
    - sentence = tokenized_review <string>
    - reviews = [sentences]

3. make vocab
    - vocab [docs, words]
        - num_doc = number of doc
        -


4. Build A
   - tfidf_result
        - find word in tfidf features

    - PMI
        - total_windows : int
        - function count(word)
            - return int
        - function count(word1, word2)
            - return int

1. Run build_graph.py
2. Run build_graph_.py