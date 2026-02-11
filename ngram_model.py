import math
from collections import defaultdict

def extract_ngrams(words, n=2):
    return ["_".join(words[i:i+n]) for i in range(len(words)-n+1)]


def train_ngram_naive_bayes(train_data, n=2):
    vocab = set()
    counts = {
        "SPORT": defaultdict(int),
        "POLITICS": defaultdict(int)
    }
    totals = defaultdict(int)
    class_counts = defaultdict(int)

    for text, label in train_data:
        class_counts[label] += 1
        words = text.lower().split()
        ngrams = extract_ngrams(words, n)

        for ng in ngrams:
            vocab.add(ng)
            counts[label][ng] += 1
            totals[label] += 1

    return vocab, counts, totals, class_counts


def predict_ngram(text, model, n=2):
    vocab, counts, totals, class_counts = model
    vocab_size = len(vocab)
    total_docs = sum(class_counts.values())
    words = text.lower().split()
    ngrams = extract_ngrams(words, n)

    scores = {}
    for label in class_counts:
        scores[label] = math.log(class_counts[label] / total_docs)
        for ng in ngrams:
            c = counts[label].get(ng, 0)
            scores[label] += math.log((c + 1) / (totals[label] + vocab_size))

    return max(scores, key=scores.get)
