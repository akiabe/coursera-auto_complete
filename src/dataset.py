def split_to_sentences(data):
    """split data by linebreak '\n' """
    sentences = data.split("\n")
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0]

    return sentences


def tokenize_sentences(sentences):
    """tokenize sentences into tokens"""
    import nltk

    tokenized_sentences = []

    for sentence in sentences:
        sentence = sentence.lower()
        tokenized = nltk.word_tokenize(sentence)
        tokenized_sentences.append(tokenized)

    return tokenized_sentences


def get_tokenized_data(data):
    """merge split_to_sentences and tokenize_sentences functions"""
    sentences = split_to_sentences(data)
    tokenized_sentences = tokenize_sentences(sentences)

    return tokenized_sentences


def split_to_train_and_test_sets(tokenized_data):
    """split tokenized sentences into train and test sets"""
    import random

    random.seed(87)
    random.shuffle(tokenized_data)

    train_size = int(len(tokenized_data) * 0.8)

    train_data = tokenized_data[0:train_size]
    test_data = tokenized_data[train_size:]

    return train_data, test_data


def count_words(tokenized_sentences):
    """count the number of word appearence in the tokenized sentences"""
    # create empty dictionary and store word to key and count number to value
    word_counts = {}

    # loop through the tokenized sentences
    for sentence in tokenized_sentences:
        # loop through the sentence
        for token in sentence:
            # if token not in dictionary, set the word and count to 1
            if token not in word_counts.keys():
                word_counts[token] = 1
            # if token is already in the dictionary, increment word count
            else:
                word_counts[token] += 1

    return word_counts


def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):
    """find the words that appear n times or more"""
    # initialize an empty list to store words that appear at least count_threshold
    closed_vocab = []

    # get the word count dictionary
    word_counts = count_words(tokenized_sentences)

    # loop through the word_counts.items()
    for word, cnt in word_counts.items():
        # if word's count is at least as great as count_threshold
        if cnt >= count_threshold:
            # append the word to the list
            closed_vocab.append(word)

    return closed_vocab


def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    """replace words not in the given vocabulary with <unk> tokens'"""
    # place vocabulary into a set for faster search
    vocabulary = set(vocabulary)

    # initialize an empty list to store hold the sentences
    replaced_tokenized_sentences = []

    # loop through the tokenized sentences
    for sentence in tokenized_sentences:
        # initialize an empty list to store a single sentence with <unk> token replacement
        replaced_sentence = []

        # loop through the sentence
        for token in sentence:
            # if token in vocabulary, append token to list
            if token in vocabulary:
                replaced_sentence.append(token)
            # otherwise, append unknown token toke list
            else:
                replaced_sentence.append(unknown_token)

        # append the list of token to the list of lists
        replaced_tokenized_sentences.append(replaced_sentence)

    return replaced_tokenized_sentences


def preprocess_data(train_data, test_data, count_threshold):
    """find tokens that appear at least n times in the training data.
       replace tokens that appear that less than N times by <unk> both for training and test data
    """
    # get the vocabulary
    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)

    # replace less common words with <unk> in train and test data
    train_data_replaced = replace_oov_words_by_unk(train_data, vocabulary, unknown_token="<unk>")
    test_data_replaced = replace_oov_words_by_unk(test_data, vocabulary, unknown_token="<unk>")

    return train_data_replaced, test_data_replaced, vocabulary
