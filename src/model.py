def count_n_grams(data, n, start_token='<s>', end_token='<e>' ):
    """count all n-gram in the data"""
    # initialize n-grams dictionary
    n_grams = {}

    # loop through the data
    for sentence in data:
        # append start and end token to sentence
        sentence = [start_token]*n + sentence + [end_token]
        # convert list to tuple and use it as a dictionary key
        sentence = tuple(sentence)

        # loop through the start to end of n-gram index
        for i in range(len(sentence) if n==1 else len(sentence)-1):
            # get n-gram from i to i+n
            n_gram = sentence[i:i+n]
            # if n-gram contain in dictionary, increment to 1
            if n_gram in n_grams.keys():
                n_grams[n_gram] += 1
            # otherwise, set 1
            else:
                n_grams[n_gram] = 1

    return n_grams


def estimate_probability(word, previous_n_gram,
                         n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    """estimate the probabilities of a next word using the n-gram counts with k-smoothing
    :param word: next word
    :param previous_n_gram: a sequence of words of length n
    :param n_gram_counts: dictionary of counts of n-grams
    :param n_plus1_gram_counts: dictionary of counts of (n+1)-grams
    :param vocabulary_size: number of words in the vocabulary
    :param k: smoothig parameter
    :return probability:
    """
    # convert list to tuple and use it as a dictionary key
    previous_n_gram = tuple(previous_n_gram)

    # set the count of n-gram in the dictionary, otherwise 0 if not in the dictionary
    previous_n_gram_count = n_gram_counts[previous_n_gram] if previous_n_gram in n_gram_counts else 0

    # calculate the denominator using the count of the previous n-gram and smoothing
    denominator = previous_n_gram_count + k*vocabulary_size

    # define n-plus1-gram as the previous n-gram plus current words as a tuple
    n_plus1_gram = previous_n_gram + (word,)

    # set the count of n-gram plus current word in the dictionary, otherwise 0 if not in the dictionary
    n_plus1_gram_counts = n_plus1_gram_counts[n_plus1_gram] if n_plus1_gram in n_plus1_gram_counts else 0

    # calculate the numerator using the count of n-gram plus current word and smoothing
    numerator = n_plus1_gram_counts + k

    # calculate the probability
    probability = numerator / denominator

    return probability


def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0):
    """estimate the probabilities of next words using the n-gram counts with k-smoothing"""
    # convert list to tuple and use it as a dictionary key
    previous_n_gram = tuple(previous_n_gram)

    # add end and unknown token to the vocabulary
    vocabulary = vocabulary + ["<e>", "<unk>"]
    vocabulary_size = len(vocabulary)

    probabilities = {}
    # loop through the vocabulary
    for word in vocabulary:
        # get the probabilities of next word
        probability = estimate_probability(word, previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=k)
        # append the probability of word to dictionary
        probabilities[word] = probability

    return probabilities


def make_count_matrix(n_plus1_gram_counts, vocabulary):
    """
    :param n_plus1_gram_counts: dictionary of counts of (n+1)-grams
    :param vocabulary: list of vocabulary
    :return count_matrix: pandas dataframe of count matrix
    """
    import numpy as np
    import pandas as pd

    # add end and unknown token to vocabulary
    vocabulary = vocabulary + ["<e>", "<unk>"]

    # obtain unique n-grams
    n_grams = []

    # loop through the keys of n-gram plus current word dictionary
    for n_plus1_gram in n_plus1_gram_counts.keys():
        n_gram = n_plus1_gram[0:-1]
        n_grams.append(n_gram)

    n_grams = list(set(n_grams))

    # mapping from n-gram to row
    row_index = {n_gram: i for i, n_gram in enumerate(n_grams)}

    # mapping from next word to column
    col_index = {word: j for j, word in enumerate(vocabulary)}

    # set number of row and column
    nrow = len(n_grams)
    ncol = len(vocabulary)

    # initialize count matrix
    count_matrix = np.zeros((nrow, ncol))

    # loop through the items of n-gram plus current word dictionary
    for n_plus1_gram, count in n_plus1_gram_counts.items():
        # get the n-gram and word from keys of n-gram plus current word dictionary
        n_gram = n_plus1_gram[0:-1]
        word = n_plus1_gram[-1]

        if word not in vocabulary:
            continue
        i = row_index[n_gram]
        j = col_index[word]
        count_matrix[i, j] = count

    # set pandas dataframe of count matrix
    count_matrix = pd.DataFrame(count_matrix,
                                index=n_grams, columns=vocabulary)

    return count_matrix


def make_probability_matrix(n_plus1_gram_counts, vocabulary, k):
    count_matrix = make_count_matrix(n_plus1_gram_counts, unique_words)
    count_matrix += k
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)

    return prob_matrix

if __name__ == "__main__":
    sentences = [['i', 'like', 'a', 'cat'],
                 ['this', 'dog', 'is', 'like', 'a', 'cat']]
    unique_words = list(set(sentences[0] + sentences[1]))
    bigram_counts = count_n_grams(sentences, 2)
    print("bigram probabilities")

    prob_matrix = make_probability_matrix(bigram_counts, unique_words, k=1)
    print(prob_matrix)