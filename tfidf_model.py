import math
from collections import defaultdict

def tokenize(text):
    return text.lower().split()


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
