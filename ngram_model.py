import math
import random
from collections import defaultdict

def tokenize(text):
    return text.lower().split()

def extract_ngrams(words, n):
    """Extract n-grams from word list.
    Example: ['I', 'love', 'sports'] with n=2 -> ['I_love', 'love_sports']
    """
    return ["_".join(words[i:i+n]) for i in range(len(words)-n+1)] if len(words) >= n else []

def extract_mixed_ngrams(words, max_n):
    """Extract mixed n-grams (1 to max_n).
    Combines unigrams, bigrams, trigrams, etc. for better coverage.
    Example: ['I', 'love', 'sports'] with max_n=2 -> ['I', 'love', 'sports', 'I_love', 'love_sports']
    """
    return [ng for n in range(1, min(max_n + 1, len(words) + 1)) 
            for ng in (words if n == 1 else extract_ngrams(words, n))]

def ngram_features(text, vocab_idx, n):
    """Convert text to n-gram feature vector."""
    ngrams = extract_ngrams(tokenize(text), n)
    features = defaultdict(int)
    for ng in ngrams:
        if ng in vocab_idx:
            features[vocab_idx[ng]] += 1
    return features

def ngram_features_mixed(text, vocab_idx, max_n):
    """Convert text to mixed n-gram feature vector."""
    ngrams = extract_mixed_ngrams(tokenize(text), max_n)
    features = defaultdict(int)
    for ng in ngrams:
        if ng in vocab_idx:
            features[vocab_idx[ng]] += 1
    return features

def cosine_similarity(vec1, vec2):
    """Cosine similarity for sparse vectors."""
    shared = set(vec1.keys()) & set(vec2.keys())
    if not shared:
        return 0.0
    dot = sum(vec1[k] * vec2[k] for k in shared)
    mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v**2 for v in vec2.values()))
    return dot / (mag1 * mag2) if mag1 and mag2 else 0.0

def sigmoid(z):
    """Sigmoid activation with overflow protection."""
    return 1.0 if z > 20 else (0.0 if z < -20 else 1 / (1 + math.exp(-z)))

# ==================== NAIVE BAYES (SINGLE N-GRAM) ====================

def train_ngram_naive_bayes(train_data, n=2):
    """Train Naive Bayes with n-grams."""
    vocab = set()  # Unique n-grams
    counts = {"SPORT": defaultdict(int), "POLITICS": defaultdict(int)}  # N-gram frequency per class
    totals = defaultdict(int)  # Total n-gram count per class
    class_counts = defaultdict(int)  # Document count per class

    # Count n-gram occurrences for each class
    for text, label in train_data:
        class_counts[label] += 1
        for ng in extract_ngrams(tokenize(text), n):
            vocab.add(ng)
            counts[label][ng] += 1
            totals[label] += 1

    return vocab, counts, totals, class_counts

def predict_ngram(text, model, n):
    """Predict using n-gram Naive Bayes."""
    vocab, counts, totals, class_counts = model
    vocab_size = len(vocab)
    total_docs = sum(class_counts.values())
    ngrams = extract_ngrams(tokenize(text), n)

    # Initialize with log prior probabilities
    scores = {label: math.log(class_counts[label] / total_docs) for label in class_counts}
    # Add log likelihood for each n-gram
    for ng in ngrams:
        for label in scores:
            c = counts[label].get(ng, 0)
            # Apply Laplace smoothing
            scores[label] += math.log((c + 1) / (totals[label] + vocab_size))

    return max(scores, key=scores.get)

# ==================== NAIVE BAYES (MIXED N-GRAMS) ====================

def train_ngram_naive_bayes_mixed(train_data, max_n=3):
    """Train Naive Bayes with mixed n-grams (1 to max_n).
    Reduces sparsity by including lower-order n-grams.
    """
    vocab = set()
    counts = {"SPORT": defaultdict(int), "POLITICS": defaultdict(int)}
    totals = defaultdict(int)
    class_counts = defaultdict(int)

    for text, label in train_data:
        class_counts[label] += 1
        for ng in extract_mixed_ngrams(tokenize(text), max_n):
            vocab.add(ng)
            counts[label][ng] += 1
            totals[label] += 1

    return vocab, counts, totals, class_counts, max_n

def predict_ngram_mixed(text, model):
    """Predict using mixed n-gram Naive Bayes."""
    vocab, counts, totals, class_counts, max_n = model
    vocab_size = len(vocab)
    total_docs = sum(class_counts.values())
    ngrams = extract_mixed_ngrams(tokenize(text), max_n)

    scores = {label: math.log(class_counts[label] / total_docs) for label in class_counts}
    for ng in ngrams:
        for label in scores:
            c = counts[label].get(ng, 0)
            scores[label] += math.log((c + 1) / (totals[label] + vocab_size))

    return max(scores, key=scores.get)

# ==================== LOGISTIC REGRESSION (SINGLE N-GRAM) ====================

def train_ngram_logistic_regression(train_data, n=2, learning_rate=0.1, epochs=20):
    """Train logistic regression with n-grams."""
    # Build n-gram vocabulary
    vocab = sorted(set(ng for text, _ in train_data for ng in extract_ngrams(tokenize(text), n)))
    vocab_idx = {ng: i for i, ng in enumerate(vocab)}
    
    # Pre-compute feature vectors and labels
    feature_vectors = []
    labels_binary = []
    label_map = {"POLITICS": 0, "SPORT": 1}
    
    for text, label in train_data:
        feature_vectors.append(ngram_features(text, vocab_idx, n))
        labels_binary.append(label_map[label])
    
    weights = [0.0] * len(vocab)
    bias = 0.0
    
    # Stochastic gradient descent
    for epoch in range(epochs):
        for idx in random.sample(range(len(train_data)), len(train_data)):
            features = feature_vectors[idx]
            y = labels_binary[idx]
            z = bias + sum(weights[i] * v for i, v in features.items())
            error = sigmoid(z) - y
            
            # Update weights
            bias -= learning_rate * error
            for i, v in features.items():
                weights[i] -= learning_rate * error * v
    
    return weights, bias, vocab_idx, n, label_map

def predict_ngram_logistic_regression(text, model):
    """Predict using n-gram logistic regression."""
    weights, bias, vocab_idx, n, label_map = model
    features = ngram_features(text, vocab_idx, n)
    z = bias + sum(weights[i] * v for i, v in features.items())
    return "SPORT" if sigmoid(z) >= 0.5 else "POLITICS"

# ==================== LOGISTIC REGRESSION (MIXED N-GRAMS) ====================

def train_ngram_logistic_regression_mixed(train_data, max_n=3, learning_rate=0.1, epochs=20):
    """Train logistic regression with mixed n-grams."""
    vocab = sorted(set(ng for text, _ in train_data for ng in extract_mixed_ngrams(tokenize(text), max_n)))
    vocab_idx = {ng: i for i, ng in enumerate(vocab)}
    
    feature_vectors = [ngram_features_mixed(text, vocab_idx, max_n) for text, _ in train_data]
    labels_binary = [0 if label == "POLITICS" else 1 for _, label in train_data]
    
    weights = [0.0] * len(vocab)
    bias = 0.0
    
    for epoch in range(epochs):
        for idx in random.sample(range(len(train_data)), len(train_data)):
            features = feature_vectors[idx]
            y = labels_binary[idx]
            z = bias + sum(weights[i] * v for i, v in features.items())
            error = sigmoid(z) - y
            
            bias -= learning_rate * error
            for i, v in features.items():
                weights[i] -= learning_rate * error * v
    
    return weights, bias, vocab_idx, max_n, {"POLITICS": 0, "SPORT": 1}

def predict_ngram_logistic_regression_mixed(text, model):
    """Predict using mixed n-gram logistic regression."""
    weights, bias, vocab_idx, max_n, label_map = model
    features = ngram_features_mixed(text, vocab_idx, max_n)
    z = bias + sum(weights[i] * v for i, v in features.items())
    return "SPORT" if sigmoid(z) >= 0.5 else "POLITICS"

# ==================== SVM (SINGLE N-GRAM) ====================

def train_ngram_svm(train_data, n=2, epochs=20, learning_rate=0.1):
    """Train SVM with n-grams."""
    vocab = sorted(set(ng for text, _ in train_data for ng in extract_ngrams(tokenize(text), n)))
    vocab_idx = {ng: i for i, ng in enumerate(vocab)}
    
    feature_vectors = [ngram_features(text, vocab_idx, n) for text, _ in train_data]
    labels_binary = [-1 if label == "POLITICS" else 1 for _, label in train_data]  # Binary labels
    
    weights = [0.0] * len(vocab)
    bias = 0.0
    
    # Perceptron: update only on misclassification
    for epoch in range(epochs):
        for idx in random.sample(range(len(train_data)), len(train_data)):
            features = feature_vectors[idx]
            y = labels_binary[idx]
            z = bias + sum(weights[i] * v for i, v in features.items())
            
            if y * z <= 0:  # Misclassified
                bias += learning_rate * y
                for i, v in features.items():
                    weights[i] += learning_rate * y * v
    
    return weights, bias, vocab_idx, n, {"POLITICS": -1, "SPORT": 1}

def predict_ngram_svm(text, model):
    """Predict using n-gram SVM."""
    weights, bias, vocab_idx, n, label_map = model
    features = ngram_features(text, vocab_idx, n)
    z = bias + sum(weights[i] * v for i, v in features.items())
    return "SPORT" if z >= 0 else "POLITICS"

# ==================== SVM (MIXED N-GRAMS) ====================

def train_ngram_svm_mixed(train_data, max_n=3, epochs=20, learning_rate=0.1):
    """Train SVM with mixed n-grams."""
    vocab = sorted(set(ng for text, _ in train_data for ng in extract_mixed_ngrams(tokenize(text), max_n)))
    vocab_idx = {ng: i for i, ng in enumerate(vocab)}
    
    feature_vectors = [ngram_features_mixed(text, vocab_idx, max_n) for text, _ in train_data]
    labels_binary = [-1 if label == "POLITICS" else 1 for _, label in train_data]
    
    weights = [0.0] * len(vocab)
    bias = 0.0
    
    for epoch in range(epochs):
        for idx in random.sample(range(len(train_data)), len(train_data)):
            features = feature_vectors[idx]
            y = labels_binary[idx]
            z = bias + sum(weights[i] * v for i, v in features.items())
            
            if y * z <= 0:
                bias += learning_rate * y
                for i, v in features.items():
                    weights[i] += learning_rate * y * v
    
    return weights, bias, vocab_idx, max_n, {"POLITICS": -1, "SPORT": 1}

def predict_ngram_svm_mixed(text, model):
    """Predict using mixed n-gram SVM."""
    weights, bias, vocab_idx, max_n, label_map = model
    features = ngram_features_mixed(text, vocab_idx, max_n)
    z = bias + sum(weights[i] * v for i, v in features.items())
    return "SPORT" if z >= 0 else "POLITICS"

# ==================== KNN (SINGLE N-GRAM) ====================

def train_ngram_knn(train_data, n=2):
    """Store training data as n-gram feature vectors."""
    vocab = sorted(set(ng for text, _ in train_data for ng in extract_ngrams(tokenize(text), n)))
    vocab_idx = {ng: i for i, ng in enumerate(vocab)}
    train_vectors = [(ngram_features(text, vocab_idx, n), label) for text, label in train_data]
    return train_vectors, vocab_idx, n

def predict_ngram_knn(text, model, k=5):
    """Predict using k-nearest neighbors with n-grams."""
    train_vectors, vocab_idx, n = model
    test_features = ngram_features(text, vocab_idx, n)
    
    # Compute similarities with all training examples
    similarities = [(cosine_similarity(test_features, vec), label) for vec, label in train_vectors]
    similarities.sort(reverse=True, key=lambda x: x[0])  # Sort by similarity
    
    # Majority vote among k nearest neighbors
    votes = defaultdict(int)
    for _, label in similarities[:k]:
        votes[label] += 1
    
    return max(votes, key=votes.get)

# ==================== KNN (MIXED N-GRAMS) ====================

def train_ngram_knn_mixed(train_data, max_n=3):
    """Store training data as mixed n-gram feature vectors."""
    vocab = sorted(set(ng for text, _ in train_data for ng in extract_mixed_ngrams(tokenize(text), max_n)))
    vocab_idx = {ng: i for i, ng in enumerate(vocab)}
    train_vectors = [(ngram_features_mixed(text, vocab_idx, max_n), label) for text, label in train_data]
    return train_vectors, vocab_idx, max_n

def predict_ngram_knn_mixed(text, model, k=5):
    """Predict using k-nearest neighbors with mixed n-grams."""
    train_vectors, vocab_idx, max_n = model
    test_features = ngram_features_mixed(text, vocab_idx, max_n)
    
    similarities = [(cosine_similarity(test_features, vec), label) for vec, label in train_vectors]
    similarities.sort(reverse=True, key=lambda x: x[0])
    
    votes = defaultdict(int)
    for _, label in similarities[:k]:
        votes[label] += 1
    
    return max(votes, key=votes.get)
