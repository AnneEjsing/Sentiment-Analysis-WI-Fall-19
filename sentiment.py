import tokeniser
from concurrent import futures
import numpy as np
from loguru import logger
import math
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.util import mark_negation
import re

t = tokeniser.Tokenizer()
stemmer = PorterStemmer()

class Model:
    def __init__(self, term_to_index, term_probability_matrix, class_prob):
        self.term_to_index = term_to_index
        self.term_probability_matrix = term_probability_matrix
        self.class_prob = class_prob
    
    def predict(self, text):
        class_scores = np.zeros((len(self.class_prob)))
        class_instances = len(self.class_prob)
        text = preprocess(text)

        # For each term calculate the probability using log. Log is used to ensure no underflow as
        # the standard formula Pi(p(x_i | c)) would be a small number
        for term in text:
            # Skip term if no occurrences in vector
            if term in self.term_to_index:
                term_index = self.term_to_index[term]
            else:
                continue

            for class_index in range(class_instances):
                class_scores[class_index] += self.term_probability_matrix[term_index][class_index]

        # Add the class probability
        for class_index in range(len(self.class_prob)):
            class_scores[class_index] += self.class_prob[class_index]

        class_scores = list(class_scores)

        # Return the class of the vector
        return class_scores.index(max(class_scores))

def load_sentiment_data(file_name):
    x, y = [], []
    current_class = None
    score_classes = {
        '0.0': None,
        '1.0': 0,
        '2.0': 0,
        '3.0': None,
        '4.0': 1,
        '5.0': 1,
    }

    with open(file_name, "r") as file:
        for line in file.readlines():
            split = [x.strip() for x in line.split(':')]

            if split[0] == 'review/score':
                current_class = score_classes[split[1]]
            elif split[0] == 'review/text' and current_class is not None:
                x.append(split[1])
                y.append(current_class)

                current_class = None

    return x, y

def generate_vocabulary(reviews):
    vocab = set()
    term_to_index = {}

    # Create vocabulary of terms
    for review in reviews:
        vocab.update([term for term in review])
    
    vocab = sorted(list(vocab))
    vocab_length = len(vocab)
    
    for i in range(vocab_length):
        term_to_index[vocab[i]] = i
        
    return vocab_length, term_to_index

def count_term_occurrence(data, labels, classes, vocab_index: dict):
    vocab_length = len(vocab_index.items())
    matrix = np.zeros((vocab_length, len(classes)))

    # For each text/review count the number of occurrences for each class
    for text, label in zip(data, labels):
        for term in text:
            # Skips if not in dictionary
            if term not in vocab_index:
                continue
            index = vocab_index[term]
            matrix[index][label] += 1

    return matrix

def calculate_term_probabilities(term_freq_matrix, terms_per_class, vocabulary_length):
    term_probability_matrix = np.zeros((len(term_freq_matrix), len(terms_per_class)))

    # Calculate the term probability for each term for a class.
    for term_index in range(len(term_freq_matrix)):
        for class_index in range(len(terms_per_class)):
            # Tricks of the trade #2 is to use log space for numerical stability
            # Laplace smoothing
            term_freq_in_class = term_freq_matrix[term_index][class_index]
            term_probability_matrix[term_index][class_index] = math.log((term_freq_in_class + 1) / (terms_per_class[class_index] + vocabulary_length))

    return term_probability_matrix

def get_correctness_percentage(predictions, labels):
    acc = len([1 for pred, label in zip(predictions, labels) if pred == label]) / len(labels)
    return acc

def preprocess(text):
    ''' Tokenises and stems the tokens of the text '''
    tokens = t.tokenize(text)
    
    #stemmed_tokens = [stemmer.stem(token) for token in tokens]
    #return stemmed_tokens

    #tokens = mark_negation(tokens)
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    return tokens

def naive_bayes():
    classes = {0, 1}
    train_x, train_y = load_sentiment_data('data/SentimentTrainingData.txt')
    logger.info('Loaded training data')

    # Preprocess train data
    train_x = [preprocess(text) for text in train_x]
    logger.info('Preprocessed training data')

    # Create vocabulary and vocabulary index
    vocabulary_length, term_to_index = generate_vocabulary(train_x)
    logger.info('Created vocabulary')

    class_prob = {}
    # Step 1
    #   Compute N
    #   Count the number of reviews
    num_reviews = len(train_y)
    logger.info('Step 1/5: Counted number of reviews')

    # Step 2
    #   Compute the probability of having a document from class c: p(c) = N(c) + 1 / N + |C|
    #   |C| = number of classes
    for cls in classes:
        num_reviews_in_class = train_y.count(cls)
        # Tricks of the trade #2 is to use log space for numerical stability
        # Laplace smoothing is applied by plussing 1 to num_reviews_in_class and len(classes) to num_reviews.
        class_prob[cls] = math.log((num_reviews_in_class + 1) / (num_reviews + len(classes)))
    logger.info('Step 2/5: Calculated class probabilities')

    # Step 3
    #   Compute N(x_i | c)
    #   Count the number of occurrences for each term over all reviews
    term_freq_matrix = count_term_occurrence(train_x, train_y, classes, term_to_index)
    logger.info('Step 3/5: Calculated term count for each class')

    # Step 4
    #   Compute N(c)
    #   Count the number of reviews with sentiment c
    terms_per_class = np.sum(term_freq_matrix, axis=0)
    logger.info('Step 4/5: Counted the number of reviews with sentiment c')

    # Step 5
    #   Compute P(x_i | c) for all possible words in corpus x_i ∈ X and all possible
    #   sentiment classes c ∈ C
    #   That is, the probability of a term occurring given a class.
    term_probability_matrix = calculate_term_probabilities(term_freq_matrix, terms_per_class, vocabulary_length)
    logger.info('Step 5/5: Calculated term probability matrix')

    return Model(term_to_index, term_probability_matrix, class_prob)

if __name__ == "__main__":
    model = naive_bayes()

    #Load test data
    test_x, test_y = load_sentiment_data('data/SentimentTestingData.txt')

    # Preprocess test data
    test_x = [preprocess(text) for text in test_x]

    # Predict the class of test labels
    predictions = []
    for text in test_x:
        predictions.append(model.predict(text))

    logger.info('Predicted classes on test set')
    
    print(get_correctness_percentage(predictions, test_y))