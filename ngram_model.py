import math
import random
from collections import defaultdict

def tokenize(text):
    return text.lower().split()


def extract_ngrams(words, n=2):
    """Extract n-grams from a list of words."""
    return ["_".join(words[i:i+n]) for i in range(len(words)-n+1)]


# ================== NAIVE BAYES (N-GRAM) ==================

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
        words = tokenize(text)
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
    words = tokenize(text)
    ngrams = extract_ngrams(words, n)

    scores = {}
    for label in class_counts:
        scores[label] = math.log(class_counts[label] / total_docs)
        for ng in ngrams:
            c = counts[label].get(ng, 0)
            scores[label] += math.log((c + 1) / (totals[label] + vocab_size))

    return max(scores, key=scores.get)


# ================== N-GRAM FEATURE EXTRACTION ==================

def ngram_features(text, vocab_idx, n=2):
    """Convert text to N-gram feature vector."""
    words = tokenize(text)
    ngrams = extract_ngrams(words, n)
    features = defaultdict(int)
    for ng in ngrams:
        if ng in vocab_idx:
            features[vocab_idx[ng]] += 1
    return features


# ================== LOGISTIC REGRESSION (N-GRAM) - OPTIMIZED ==================

def sigmoid(z):
    """Sigmoid activation function."""
    if z > 20:
        return 1.0
    elif z < -20:
        return 0.0
    return 1 / (1 + math.exp(-z))


def train_ngram_logistic_regression(train_data, n=2, learning_rate=0.1, epochs=20):
    """Train logistic regression with N-gram features - OPTIMIZED."""
    # Build vocabulary of n-grams
    vocab = set()
    for text, _ in train_data:
        words = tokenize(text)
        ngrams = extract_ngrams(words, n)
        vocab.update(ngrams)
    
    vocab = sorted(list(vocab))
    vocab_idx = {ng: i for i, ng in enumerate(vocab)}
    
    # Pre-compute feature vectors
    feature_vectors = []
    labels_binary = []
    label_map = {"POLITICS": 0, "SPORT": 1}
    
    for text, label in train_data:
        features = ngram_features(text, vocab_idx, n)
        feature_vectors.append(features)
        labels_binary.append(label_map[label])
    
    # Initialize weights
    weights = [0.0] * len(vocab)
    bias = 0.0
    
    # Training
    for epoch in range(epochs):
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        
        for idx in indices:
            features = feature_vectors[idx]
            y = labels_binary[idx]
            
            # Prediction (sparse computation)
            z = bias
            for feat_idx, feat_val in features.items():
                z += weights[feat_idx] * feat_val
            
            pred = sigmoid(z)
            error = pred - y
            
            # Update (sparse update)
            bias -= learning_rate * error
            for feat_idx, feat_val in features.items():
                weights[feat_idx] -= learning_rate * error * feat_val
    
    return weights, bias, vocab_idx, n, label_map


def predict_ngram_logistic_regression(text, model):
    """Predict using trained logistic regression model."""
    weights, bias, vocab_idx, n, label_map = model
    features = ngram_features(text, vocab_idx, n)
    
    z = bias
    for feat_idx, feat_val in features.items():
        z += weights[feat_idx] * feat_val
    
    pred = sigmoid(z)
    
    # Return label based on threshold 0.5
    if pred >= 0.5:
        return "SPORT"
    else:
        return "POLITICS"


# ================== SVM PERCEPTRON (N-GRAM) - OPTIMIZED ==================

def train_ngram_svm(train_data, n=2, epochs=20, learning_rate=0.1):
    """Train SVM using perceptron algorithm with N-gram features - OPTIMIZED."""
    # Build vocabulary of n-grams
    vocab = set()
    for text, _ in train_data:
        words = tokenize(text)
        ngrams = extract_ngrams(words, n)
        vocab.update(ngrams)
    
    vocab = sorted(list(vocab))
    vocab_idx = {ng: i for i, ng in enumerate(vocab)}
    
    # Pre-compute feature vectors
    feature_vectors = []
    labels_binary = []
    label_map = {"POLITICS": -1, "SPORT": 1}
    
    for text, label in train_data:
        features = ngram_features(text, vocab_idx, n)
        feature_vectors.append(features)
        labels_binary.append(label_map[label])
    
    # Initialize weights
    weights = [0.0] * len(vocab)
    bias = 0.0
    
    # Training
    for epoch in range(epochs):
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        
        for idx in indices:
            features = feature_vectors[idx]
            y = labels_binary[idx]
            
            # Prediction (sparse computation)
            z = bias
            for feat_idx, feat_val in features.items():
                z += weights[feat_idx] * feat_val
            
            # Update if misclassified
            if y * z <= 0:
                bias += learning_rate * y
                for feat_idx, feat_val in features.items():
                    weights[feat_idx] += learning_rate * y * feat_val
    
    return weights, bias, vocab_idx, n, label_map


def predict_ngram_svm(text, model):
    """Predict using trained SVM model."""
    weights, bias, vocab_idx, n, label_map = model
    features = ngram_features(text, vocab_idx, n)
    
    z = bias
    for feat_idx, feat_val in features.items():
        z += weights[feat_idx] * feat_val
    
    if z >= 0:
        return "SPORT"
    else:
        return "POLITICS"


# ================== K-NEAREST NEIGHBORS (N-GRAM) - OPTIMIZED ==================

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two sparse vectors - OPTIMIZED."""
    shared_keys = set(vec1.keys()) & set(vec2.keys())
    if not shared_keys:
        return 0.0
    
    dot_product = sum(vec1[k] * vec2[k] for k in shared_keys)
    
    mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v**2 for v in vec2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)


def train_ngram_knn(train_data, n=2):
    """KNN doesn't have a training phase - just store the data."""
    vocab = set()
    for text, _ in train_data:
        words = tokenize(text)
        ngrams = extract_ngrams(words, n)
        vocab.update(ngrams)
    
    vocab = sorted(list(vocab))
    vocab_idx = {ng: i for i, ng in enumerate(vocab)}
    
    # Convert all training data to feature vectors
    train_vectors = []
    for text, label in train_data:
        features = ngram_features(text, vocab_idx, n)
        train_vectors.append((features, label))
    
    return train_vectors, vocab_idx, n


def predict_ngram_knn(text, model, k=5):
    """Predict using KNN with cosine similarity."""
    train_vectors, vocab_idx, n = model
    test_features = ngram_features(text, vocab_idx, n)
    
    # Calculate similarity with all training examples
    similarities = []
    for train_features, label in train_vectors:
        sim = cosine_similarity(test_features, train_features)
        similarities.append((sim, label))
    
    # Get top k neighbors using sort
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_k = similarities[:k]
    
    # Vote
    votes = defaultdict(int)
    for _, label in top_k:
        votes[label] += 1
    
    return max(votes, key=votes.get)