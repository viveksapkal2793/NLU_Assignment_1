import math
import random
from collections import defaultdict

def tokenize(text):
    return text.lower().split()

def tfidf_features(text, vocab_idx, df, N):
    """Convert text to TF-IDF feature vector.
    TF-IDF = Term Frequency × Inverse Document Frequency
    TF-IDF(word) = count(word) × log(N / DF(word))
    Weights rare words higher, common words lower.
    """
    counts = defaultdict(int)
    for w in tokenize(text):
        counts[w] += 1
    
    features = {}
    for w, c in counts.items():
        if w in df and w in vocab_idx:
            # TF = c (term frequency), IDF = log(N / (1 + DF))
            features[vocab_idx[w]] = c * math.log(N / (1 + df[w]))
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

def train_tfidf_naive_bayes(train_data):
    """Train Naive Bayes using TF-IDF features."""
    df = defaultdict(int)  # Document frequency: number of docs containing each word
    tf = []  # Term frequency: word counts per document
    labels = []

    # Compute term frequencies and document frequencies
    for text, label in train_data:
        labels.append(label)
        counts = defaultdict(int)
        for w in tokenize(text):
            counts[w] += 1
        tf.append(counts)
        for w in counts:
            df[w] += 1  # Increment document frequency

    N = len(train_data)
    # Compute TF-IDF vectors for all training documents
    tfidf = []
    for counts in tf:
        vec = {w: c * math.log(N / (1 + df[w])) for w, c in counts.items()}
        tfidf.append(vec)

    return tfidf, labels, df, N

def predict_tfidf(text, model):
    """Predict using TF-IDF Naive Bayes (class centroids).
    Uses dot product similarity between test document and class centroids.
    """
    tfidf, labels, df, N = model
    
    # Compute TF-IDF for test document
    test_counts = defaultdict(int)
    for w in tokenize(text):
        test_counts[w] += 1
    test_tfidf = {w: c * math.log(N / (1 + df[w])) for w, c in test_counts.items() if w in df}
    
    # Compute similarity scores with each class
    class_scores = defaultdict(float)
    class_doc_count = defaultdict(int)
    
    for i, vec in enumerate(tfidf):
        label = labels[i]
        class_doc_count[label] += 1
        # Dot product between test and training document
        for w in test_tfidf:
            if w in vec:
                class_scores[label] += test_tfidf[w] * vec[w]
    
    # Average scores by class size (centroid approach)
    for label in class_scores:
        class_scores[label] /= class_doc_count[label]
    
    return max(class_scores, key=class_scores.get)

# ==================== LOGISTIC REGRESSION ====================

def train_tfidf_logistic_regression(train_data, learning_rate=0.1, epochs=20):
    """Train logistic regression with TF-IDF features."""
    # Compute document frequency for all words
    df = defaultdict(int)
    for text, _ in train_data:
        for w in set(tokenize(text)):
            df[w] += 1
    
    N = len(train_data)
    vocab = sorted(df.keys())
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    
    # Pre-compute TF-IDF features for efficiency
    feature_vectors = [tfidf_features(text, vocab_idx, df, N) for text, _ in train_data]
    labels_binary = [0 if label == "POLITICS" else 1 for _, label in train_data]
    
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
    
    return weights, bias, vocab_idx, df, N, {"POLITICS": 0, "SPORT": 1}

def predict_tfidf_logistic_regression(text, model):
    """Predict using TF-IDF logistic regression."""
    weights, bias, vocab_idx, df, N, label_map = model
    features = tfidf_features(text, vocab_idx, df, N)
    z = bias + sum(weights[i] * v for i, v in features.items())
    return "SPORT" if sigmoid(z) >= 0.5 else "POLITICS"

# ==================== SVM ====================

def train_tfidf_svm(train_data, epochs=20, learning_rate=0.1):
    """Train SVM with TF-IDF features."""
    # Compute document frequency
    df = defaultdict(int)
    for text, _ in train_data:
        for w in set(tokenize(text)):
            df[w] += 1
    
    N = len(train_data)
    vocab = sorted(df.keys())
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    
    feature_vectors = [tfidf_features(text, vocab_idx, df, N) for text, _ in train_data]
    labels_binary = [-1 if label == "POLITICS" else 1 for _, label in train_data]  # Binary labels
    
    weights = [0.0] * len(vocab)
    bias = 0.0
    
    # Perceptron algorithm: update on misclassification
    for epoch in range(epochs):
        for idx in random.sample(range(len(train_data)), len(train_data)):
            features = feature_vectors[idx]
            y = labels_binary[idx]
            z = bias + sum(weights[i] * v for i, v in features.items())
            
            if y * z <= 0:  # Misclassified
                bias += learning_rate * y
                for i, v in features.items():
                    weights[i] += learning_rate * y * v
    
    return weights, bias, vocab_idx, df, N, {"POLITICS": -1, "SPORT": 1}

def predict_tfidf_svm(text, model):
    """Predict using TF-IDF SVM."""
    weights, bias, vocab_idx, df, N, label_map = model
    features = tfidf_features(text, vocab_idx, df, N)
    z = bias + sum(weights[i] * v for i, v in features.items())
    return "SPORT" if z >= 0 else "POLITICS"

# ==================== KNN ====================

def train_tfidf_knn(train_data):
    """Store training data as TF-IDF feature vectors."""
    df = defaultdict(int)
    for text, _ in train_data:
        for w in set(tokenize(text)):
            df[w] += 1
    
    N = len(train_data)
    vocab = sorted(df.keys())
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    train_vectors = [(tfidf_features(text, vocab_idx, df, N), label) for text, label in train_data]
    
    return train_vectors, vocab_idx, df, N

def predict_tfidf_knn(text, model, k=5):
    """Predict using k-nearest neighbors with TF-IDF."""
    train_vectors, vocab_idx, df, N = model
    test_features = tfidf_features(text, vocab_idx, df, N)
    
    # Compute cosine similarity with all training documents
    similarities = [(cosine_similarity(test_features, vec), label) for vec, label in train_vectors]
    similarities.sort(reverse=True, key=lambda x: x[0])  # Sort by similarity
    
    # Majority vote among k nearest neighbors
    votes = defaultdict(int)
    for _, label in similarities[:k]:
        votes[label] += 1
    
    return max(votes, key=votes.get)
