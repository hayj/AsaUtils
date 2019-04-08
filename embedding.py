import numpy as np
import copy


def loadWordVectors(path=None, maxWords=None):
    """
        This function load a file containing vector.
        At each line the first token must be the word.
    """
    if path is None:
        path = nosaveDir() + "/WordVectors/fasttext/crawl-300d-2M.vec"
    file = open(path, "r")
    vectors = {}
    for line in file.readlines():
        if maxWords is not None and len(vectors) == maxWords:
            break
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        vectors[word] = vector
    return vectors


def tokensToEmbedding(tokens, wordVectors, operation='sum', removeDuplicates=True, doLower=False):
    """
        This function take tokens (or a list of tokens)
        And a map word->vector
        It return a sentence embedding according to the operation given (sum, mean).
    """
    if isinstance(tokens[0], list):
        tokens = copy.deepcopy(tokens)
        for i in range(len(tokens)):
            tokens[i] = tokensToEmbedding(tokens[i], wordVectors, operation=operation,
                                          removeDuplicates=removeDuplicates, doLower=doLower)
        return tokens
    else:
        if removeDuplicates:
            tokens = set(tokens)
        vectors = []
        for current in tokens:
            if lower:
                current = current.lower()
            if current in wordVectors:
                vectors.append(wordVectors["current"])
        if operation == 'sum':
            return np.sum(np.array(vectors), axis=0)
        elif operation == 'mean':
            return np.mean(np.array(vectors), axis=0)
        return vectors