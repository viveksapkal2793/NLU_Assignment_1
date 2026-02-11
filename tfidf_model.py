import math
import random
from collections import defaultdict

def tokenize(text):
    return text.lower().split()


# ================== NAIVE BAYES (TF-IDF) ==================

def train_tfidf_naive_bayes(train_data):
    df = defaultdict(int)
    tf = []
    labels = []

    for text, label in train_data:
        labels.append(label)
        words = tokenize(text)
        counts = defaultdict(int)
        for w in words:
            counts[w] += 1
        tf.append(counts)
        for w in counts:
            df[w] += 1

    N = len(train_data)
    tfidf = []
    for counts in tf:
        vec = {}
        for w, c in counts.items():
            vec[w] = c * math.log(N / (1 + df[w]))
        tfidf.append(vec)

    return tfidf, labels, df, N


def predict_tfidf(text, model):
    tfidf, labels, df, N = model
    words = tokenize(text)
    
    # Create TF-IDF vector for test document
    test_counts = defaultdict(int)
    for w in words:
        test_counts[w] += 1
    
    test_tfidf = {}
    for w, c in test_counts.items():
        if w in df:
            test_tfidf[w] = c * math.log(N / (1 + df[w]))
    
    # Calculate cosine similarity or dot product with class centroids
    class_scores = defaultdict(float)
    class_doc_count = defaultdict(int)
    
    for i, vec in enumerate(tfidf):
        label = labels[i]
        class_doc_count[label] += 1
        for w in test_tfidf:
            if w in vec:
                class_scores[label] += test_tfidf[w] * vec[w]
    
    # Average by class size
    for label in class_scores:
        class_scores[label] /= class_doc_count[label]
    
    return max(class_scores, key=class_scores.get)


# ================== TF-IDF FEATURE EXTRACTION ==================

def tfidf_features(text, vocab_idx, df, N):
    """Convert text to TF-IDF feature vector."""
    words = tokenize(text)
    counts = defaultdict(int)
    for w in words:
        counts[w] += 1
    
    features = {}
    for w, c in counts.items():
        if w in df:
            idx = vocab_idx.get(w)
            if idx is not None:
                features[idx] = c * math.log(N / (1 + df[w]))
    return features


# ================== LOGISTIC REGRESSION (TF-IDF) - OPTIMIZED ==================

def sigmoid(z):
    """Sigmoid activation function."""
    if z > 20:
        return 1.0
    elif z < -20:
        return 0.0
    return 1 / (1 + math.exp(-z))


def train_tfidf_logistic_regression(train_data, learning_rate=0.1, epochs=20):
    """Train logistic regression with TF-IDF features - OPTIMIZED."""
    # First, compute document frequency
    df = defaultdict(int)
    for text, _ in train_data:
        words = set(tokenize(text))
        for w in words:
            df[w] += 1
    
    N = len(train_data)
    
    # Build vocabulary
    vocab = sorted(list(df.keys()))
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    
    # Pre-compute feature vectors
    feature_vectors = []
    labels_binary = []
    label_map = {"POLITICS": 0, "SPORT": 1}
    
    for text, label in train_data:
        features = tfidf_features(text, vocab_idx, df, N)
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
    
    return weights, bias, vocab_idx, df, N, label_map


def predict_tfidf_logistic_regression(text, model):
    """Predict using trained logistic regression model."""
    weights, bias, vocab_idx, df, N, label_map = model
    features = tfidf_features(text, vocab_idx, df, N)
    
    z = bias
    for feat_idx, feat_val in features.items():
        z += weights[feat_idx] * feat_val
    
    pred = sigmoid(z)
    
    if pred >= 0.5:
        return "SPORT"
    else:
        return "POLITICS"


# ================== SVM PERCEPTRON (TF-IDF) - OPTIMIZED ==================

def train_tfidf_svm(train_data, epochs=20, learning_rate=0.1):
    """Train SVM using perceptron algorithm with TF-IDF features - OPTIMIZED."""
    # Compute document frequency
    df = defaultdict(int)
    for text, _ in train_data:
        words = set(tokenize(text))
        for w in words:
            df[w] += 1
    
    N = len(train_data)
    vocab = sorted(list(df.keys()))
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    
    # Pre-compute features
    feature_vectors = []
    labels_binary = []
    label_map = {"POLITICS": -1, "SPORT": 1}
    
    for text, label in train_data:
        features = tfidf_features(text, vocab_idx, df, N)
        feature_vectors.append(features)
        labels_binary.append(label_map[label])
    
    weights = [0.0] * len(vocab)
    bias = 0.0
    
    for epoch in range(epochs):
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        
        for idx in indices:
            features = feature_vectors[idx]
            y = labels_binary[idx]
            
            z = bias
            for feat_idx, feat_val in features.items():
                z += weights[feat_idx] * feat_val
            
            if y * z <= 0:
                bias += learning_rate * y
                for feat_idx, feat_val in features.items():
                    weights[feat_idx] += learning_rate * y * feat_val
    
    return weights, bias, vocab_idx, df, N, label_map


def predict_tfidf_svm(text, model):
    """Predict using trained SVM model."""
    weights, bias, vocab_idx, df, N, label_map = model
    features = tfidf_features(text, vocab_idx, df, N)
    
    z = bias
    for feat_idx, feat_val in features.items():
        z += weights[feat_idx] * feat_val
    
    if z >= 0:
        return "SPORT"
    else:
        return "POLITICS"


# ================== K-NEAREST NEIGHBORS (TF-IDF) ==================

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


def train_tfidf_knn(train_data):
    """KNN doesn't have a training phase - just store the data."""
    df = defaultdict(int)
    for text, _ in train_data:
        words = set(tokenize(text))
        for w in words:
            df[w] += 1
    
    N = len(train_data)
    vocab = sorted(list(df.keys()))
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    
    train_vectors = []
    for text, label in train_data:
        features = tfidf_features(text, vocab_idx, df, N)
        train_vectors.append((features, label))
    
    return train_vectors, vocab_idx, df, N


def predict_tfidf_knn(text, model, k=5):
    """Predict using KNN with cosine similarity."""
    train_vectors, vocab_idx, df, N = model
    test_features = tfidf_features(text, vocab_idx, df, N)
    
    similarities = []
    for train_features, label in train_vectors:
        sim = cosine_similarity(test_features, train_features)
        similarities.append((sim, label))
    
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_k = similarities[:k]
    
    votes = defaultdict(int)
    for _, label in top_k:
        votes[label] += 1
    
    return max(votes, key=votes.get)