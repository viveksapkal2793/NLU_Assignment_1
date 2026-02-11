import math
import random
from collections import defaultdict

def tokenize(text):
    return text.lower().split()

def bow_features(text, vocab_idx):
    """Convert text to sparse BoW feature vector."""
    features = defaultdict(int)
    for word in tokenize(text):
        if word in vocab_idx:
            features[vocab_idx[word]] += 1
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

# ==================== NAIVE BAYES ====================

def train_bow_naive_bayes(train_data):
    """Train Naive Bayes with Laplace smoothing."""
    vocab = set()
    word_counts = {"SPORT": defaultdict(int), "POLITICS": defaultdict(int)}
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
    """Predict class using Naive Bayes."""
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

# ==================== LOGISTIC REGRESSION ====================

def train_bow_logistic_regression(train_data, learning_rate=0.1, epochs=20):
    """Train logistic regression with stochastic gradient descent."""
    vocab = sorted(set(word for text, _ in train_data for word in tokenize(text)))
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    
    # Pre-compute features
    feature_vectors = []
    labels_binary = []
    label_map = {"POLITICS": 0, "SPORT": 1}
    
    for text, label in train_data:
        feature_vectors.append(bow_features(text, vocab_idx))
        labels_binary.append(label_map[label])
    
    # Initialize weights
    weights = [0.0] * len(vocab)
    bias = 0.0
    
    # SGD training
    for epoch in range(epochs):
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        
        for idx in indices:
            features = feature_vectors[idx]
            y = labels_binary[idx]
            
            # Forward pass
            z = bias + sum(weights[i] * v for i, v in features.items())
            pred = sigmoid(z)
            error = pred - y
            
            # Backward pass
            bias -= learning_rate * error
            for i, v in features.items():
                weights[i] -= learning_rate * error * v
    
    return weights, bias, vocab_idx, label_map

def predict_bow_logistic_regression(text, model):
    """Predict using logistic regression."""
    weights, bias, vocab_idx, label_map = model
    features = bow_features(text, vocab_idx)
    z = bias + sum(weights[i] * v for i, v in features.items())
    return "SPORT" if sigmoid(z) >= 0.5 else "POLITICS"

# ==================== SVM ====================

def train_bow_svm(train_data, epochs=20, learning_rate=0.1):
    """Train SVM using perceptron algorithm."""
    vocab = sorted(set(word for text, _ in train_data for word in tokenize(text)))
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    
    # Pre-compute features
    feature_vectors = []
    labels_binary = []
    label_map = {"POLITICS": -1, "SPORT": 1}
    
    for text, label in train_data:
        feature_vectors.append(bow_features(text, vocab_idx))
        labels_binary.append(label_map[label])
    
    weights = [0.0] * len(vocab)
    bias = 0.0
    
    # Perceptron training
    for epoch in range(epochs):
        for idx in random.sample(range(len(train_data)), len(train_data)):
            features = feature_vectors[idx]
            y = labels_binary[idx]
            z = bias + sum(weights[i] * v for i, v in features.items())
            
            # Update on misclassification
            if y * z <= 0:
                bias += learning_rate * y
                for i, v in features.items():
                    weights[i] += learning_rate * y * v
    
    return weights, bias, vocab_idx, label_map

def predict_bow_svm(text, model):
    """Predict using SVM."""
    weights, bias, vocab_idx, label_map = model
    features = bow_features(text, vocab_idx)
    z = bias + sum(weights[i] * v for i, v in features.items())
    return "SPORT" if z >= 0 else "POLITICS"

# ==================== KNN ====================

def train_bow_knn(train_data):
    """Store training data as feature vectors."""
    vocab = sorted(set(word for text, _ in train_data for word in tokenize(text)))
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    train_vectors = [(bow_features(text, vocab_idx), label) for text, label in train_data]
    return train_vectors, vocab_idx

def predict_bow_knn(text, model, k=5):
    """Predict using k-nearest neighbors."""
    train_vectors, vocab_idx = model
    test_features = bow_features(text, vocab_idx)
    
    # Compute similarities and get top k
    similarities = [(cosine_similarity(test_features, vec), label) for vec, label in train_vectors]
    similarities.sort(reverse=True, key=lambda x: x[0])
    
    # Majority vote
    votes = defaultdict(int)
    for _, label in similarities[:k]:
        votes[label] += 1
    
    return max(votes, key=votes.get)
