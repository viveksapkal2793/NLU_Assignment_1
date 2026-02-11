# bow_model.py
import math
from collections import defaultdict

def tokenize(text):
    return text.lower().split()


def train_bow_naive_bayes(train_data):
    vocab = set()
    word_counts = {
        "SPORT": defaultdict(int),
        "POLITICS": defaultdict(int)
    }
    class_counts = defaultdict(int)
    total_words = defaultdict(int)

    for text, label in train_data:
        class_counts[label] += 1
        for word in tokenize(text):
            vocab.add(word)
            word_counts[label][word] += 1
            total_words[label] += 1

    return vocab, word_counts, total_words, class_counts


def predict_bow(text, model):
    vocab, word_counts, total_words, class_counts = model
    vocab_size = len(vocab)
    total_docs = sum(class_counts.values())

    scores = {}
    for label in class_counts:
        scores[label] = math.log(class_counts[label] / total_docs)

        for word in tokenize(text):
            count = word_counts[label].get(word, 0)
            scores[label] += math.log((count + 1) / (total_words[label] + vocab_size))

    return max(scores, key=scores.get)
