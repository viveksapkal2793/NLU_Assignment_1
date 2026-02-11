import math
import random
from collections import defaultdict

def tokenize(text):
    return text.lower().split()


# ================== NAIVE BAYES (BOW) ==================

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


# ================== BOW FEATURE EXTRACTION ==================

def bow_features(text, vocab_idx):
    """Convert text to BoW feature vector."""
    words = tokenize(text)
    features = defaultdict(int)
    for word in words:
        if word in vocab_idx:
            features[vocab_idx[word]] += 1
    return features


# ================== LOGISTIC REGRESSION (BOW) - OPTIMIZED ==================

def sigmoid(z):
    """Sigmoid activation function."""
    if z > 20:
        return 1.0
    elif z < -20:
        return 0.0
    return 1 / (1 + math.exp(-z))


def train_bow_logistic_regression(train_data, learning_rate=0.1, epochs=20):
    """Train logistic regression with BoW features - OPTIMIZED."""
    # Build vocabulary
    vocab = set()
    for text, _ in train_data:
        vocab.update(tokenize(text))
    
    vocab = sorted(list(vocab))
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    
    # Pre-compute feature vectors to avoid repeated tokenization
    feature_vectors = []
    labels_binary = []
    label_map = {"POLITICS": 0, "SPORT": 1}
    
    for text, label in train_data:
        features = bow_features(text, vocab_idx)
        feature_vectors.append(features)
        labels_binary.append(label_map[label])
    
    # Initialize weights
    weights = [0.0] * len(vocab)
    bias = 0.0
    
    # Training with mini-batch updates
    for epoch in range(epochs):
        # Shuffle indices
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        
        for idx in indices:
            features = feature_vectors[idx]
            y = labels_binary[idx]
            
            # Prediction (only compute for non-zero features)
            z = bias
            for feat_idx, feat_val in features.items():
                z += weights[feat_idx] * feat_val
            
            pred = sigmoid(z)
            error = pred - y
            
            # Update (only update non-zero features)
            bias -= learning_rate * error
            for feat_idx, feat_val in features.items():
                weights[feat_idx] -= learning_rate * error * feat_val
    
    return weights, bias, vocab_idx, label_map


def predict_bow_logistic_regression(text, model):
    """Predict using trained logistic regression model."""
    weights, bias, vocab_idx, label_map = model
    features = bow_features(text, vocab_idx)
    
    z = bias
    for feat_idx, feat_val in features.items():
        z += weights[feat_idx] * feat_val
    
    pred = sigmoid(z)
    
    if pred >= 0.5:
        return "SPORT"
    else:
        return "POLITICS"


# ================== SVM PERCEPTRON (BOW) - OPTIMIZED ==================

def train_bow_svm(train_data, epochs=20, learning_rate=0.1):
    """Train SVM using perceptron algorithm with BoW features - OPTIMIZED."""
    # Build vocabulary
    vocab = set()
    for text, _ in train_data:
        vocab.update(tokenize(text))
    
    vocab = sorted(list(vocab))
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    
    # Pre-compute feature vectors
    feature_vectors = []
    labels_binary = []
    label_map = {"POLITICS": -1, "SPORT": 1}
    
    for text, label in train_data:
        features = bow_features(text, vocab_idx)
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
    
    return weights, bias, vocab_idx, label_map


def predict_bow_svm(text, model):
    """Predict using trained SVM model."""
    weights, bias, vocab_idx, label_map = model
    features = bow_features(text, vocab_idx)
    
    z = bias
    for feat_idx, feat_val in features.items():
        z += weights[feat_idx] * feat_val
    
    if z >= 0:
        return "SPORT"
    else:
        return "POLITICS"


# ================== K-NEAREST NEIGHBORS (BOW) ==================

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two sparse vectors."""
    # Only compute for shared indices
    shared_keys = set(vec1.keys()) & set(vec2.keys())
    if not shared_keys:
        return 0.0
    
    dot_product = sum(vec1[k] * vec2[k] for k in shared_keys)
    
    mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v**2 for v in vec2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)


def train_bow_knn(train_data):
    """KNN doesn't have a training phase - just store the data."""
    vocab = set()
    for text, _ in train_data:
        vocab.update(tokenize(text))
    
    vocab = sorted(list(vocab))
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    
    # Convert all training data to feature vectors
    train_vectors = []
    for text, label in train_data:
        features = bow_features(text, vocab_idx)
        train_vectors.append((features, label))
    
    return train_vectors, vocab_idx


def predict_bow_knn(text, model, k=5):
    """Predict using KNN with cosine similarity."""
    train_vectors, vocab_idx = model
    test_features = bow_features(text, vocab_idx)
    
    # Calculate similarity with all training examples
    similarities = []
    for train_features, label in train_vectors:
        sim = cosine_similarity(test_features, train_features)
        similarities.append((sim, label))
    
    # Get top k neighbors using partial sort (faster than full sort)
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_k = similarities[:k]
    
    # Vote
    votes = defaultdict(int)
    for _, label in top_k:
        votes[label] += 1
    
    return max(votes, key=votes.get)