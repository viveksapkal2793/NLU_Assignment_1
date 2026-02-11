from utils import load_bbc_data, train_test_split
from data_analysis import analyze_dataset
from bow_model import (
    train_bow_naive_bayes, predict_bow,
    train_bow_logistic_regression, predict_bow_logistic_regression,
    train_bow_svm, predict_bow_svm,
    train_bow_knn, predict_bow_knn
)
from ngram_model import (
    train_ngram_naive_bayes, predict_ngram,
    train_ngram_logistic_regression, predict_ngram_logistic_regression,
    train_ngram_svm, predict_ngram_svm,
    train_ngram_knn, predict_ngram_knn
)
from tfidf_model import (
    train_tfidf_naive_bayes, predict_tfidf,
    train_tfidf_logistic_regression, predict_tfidf_logistic_regression,
    train_tfidf_svm, predict_tfidf_svm,
    train_tfidf_knn, predict_tfidf_knn
)


def evaluate(test_data, predict_fn, model):
    correct = 0
    for text, label in test_data:
        if predict_fn(text, model) == label:
            correct += 1
    return correct / len(test_data)


def main():
    # Load and analyze data
    data = load_bbc_data("bbc-text.csv")
    stats = analyze_dataset(data)
    
    # Split data
    train_data, test_data = train_test_split(data)
    
    print("=" * 70)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 70)
    print(f"\nTraining samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}\n")

    # Define n-gram sizes to test
    ngram_sizes = [2, 3, 4]  # Bigram, Trigram, Quadgram
    
    # Store results for summary
    results = {
        'bow': {},
        'ngrams': {n: {} for n in ngram_sizes},
        'tfidf': {}
    }

    # ==================== NAIVE BAYES ====================
    print("=" * 70)
    print("NAIVE BAYES MODELS")
    print("=" * 70)

    # Bag of Words
    bow_model = train_bow_naive_bayes(train_data)
    bow_acc = evaluate(test_data, predict_bow, bow_model)
    results['bow']['nb'] = bow_acc
    print(f"Bag of Words         : {bow_acc:.4f}")

    # N-grams (multiple sizes)
    for n in ngram_sizes:
        ngram_model = train_ngram_naive_bayes(train_data, n=n)
        ngram_acc = evaluate(test_data, lambda t, m: predict_ngram(t, m, n), ngram_model)
        results['ngrams'][n]['nb'] = ngram_acc
        print(f"N-grams ({n})          : {ngram_acc:.4f}")

    # TF-IDF
    tfidf_model = train_tfidf_naive_bayes(train_data)
    tfidf_acc = evaluate(test_data, predict_tfidf, tfidf_model)
    results['tfidf']['nb'] = tfidf_acc
    print(f"TF-IDF               : {tfidf_acc:.4f}")

    # ==================== LOGISTIC REGRESSION ====================
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION")
    print("=" * 70)

    # BoW
    lr_bow_model = train_bow_logistic_regression(train_data, epochs=20)
    lr_bow_acc = evaluate(test_data, predict_bow_logistic_regression, lr_bow_model)
    results['bow']['lr'] = lr_bow_acc
    print(f"BoW                  : {lr_bow_acc:.4f}")

    # N-grams (multiple sizes)
    for n in ngram_sizes:
        lr_ngram_model = train_ngram_logistic_regression(train_data, n=n, epochs=20)
        lr_ngram_acc = evaluate(test_data, predict_ngram_logistic_regression, lr_ngram_model)
        results['ngrams'][n]['lr'] = lr_ngram_acc
        print(f"N-grams ({n})          : {lr_ngram_acc:.4f}")

    # TF-IDF
    lr_tfidf_model = train_tfidf_logistic_regression(train_data, epochs=20)
    lr_tfidf_acc = evaluate(test_data, predict_tfidf_logistic_regression, lr_tfidf_model)
    results['tfidf']['lr'] = lr_tfidf_acc
    print(f"TF-IDF               : {lr_tfidf_acc:.4f}")

    # ==================== SVM ====================
    print("\n" + "=" * 70)
    print("SVM (PERCEPTRON)")
    print("=" * 70)

    # BoW
    svm_bow_model = train_bow_svm(train_data, epochs=20)
    svm_bow_acc = evaluate(test_data, predict_bow_svm, svm_bow_model)
    results['bow']['svm'] = svm_bow_acc
    print(f"BoW                  : {svm_bow_acc:.4f}")

    # N-grams (multiple sizes)
    for n in ngram_sizes:
        svm_ngram_model = train_ngram_svm(train_data, n=n, epochs=20)
        svm_ngram_acc = evaluate(test_data, predict_ngram_svm, svm_ngram_model)
        results['ngrams'][n]['svm'] = svm_ngram_acc
        print(f"N-grams ({n})          : {svm_ngram_acc:.4f}")

    # TF-IDF
    svm_tfidf_model = train_tfidf_svm(train_data, epochs=20)
    svm_tfidf_acc = evaluate(test_data, predict_tfidf_svm, svm_tfidf_model)
    results['tfidf']['svm'] = svm_tfidf_acc
    print(f"TF-IDF               : {svm_tfidf_acc:.4f}")

    # ==================== KNN ====================
    print("\n" + "=" * 70)
    print("K-NEAREST NEIGHBORS (k=5)")
    print("=" * 70)

    # BoW
    knn_bow_model = train_bow_knn(train_data)
    knn_bow_acc = evaluate(test_data, lambda t, m: predict_bow_knn(t, m, k=5), knn_bow_model)
    results['bow']['knn'] = knn_bow_acc
    print(f"BoW                  : {knn_bow_acc:.4f}")

    # N-grams (multiple sizes)
    for n in ngram_sizes:
        knn_ngram_model = train_ngram_knn(train_data, n=n)
        knn_ngram_acc = evaluate(test_data, lambda t, m: predict_ngram_knn(t, m, k=5), knn_ngram_model)
        results['ngrams'][n]['knn'] = knn_ngram_acc
        print(f"N-grams ({n})          : {knn_ngram_acc:.4f}")

    # TF-IDF
    knn_tfidf_model = train_tfidf_knn(train_data)
    knn_tfidf_acc = evaluate(test_data, lambda t, m: predict_tfidf_knn(t, m, k=5), knn_tfidf_model)
    results['tfidf']['knn'] = knn_tfidf_acc
    print(f"TF-IDF               : {knn_tfidf_acc:.4f}")

    # ==================== SUMMARY ====================
    print("\n" + "=" * 70)
    print("SUMMARY - COMPLETE COMPARISON")
    print("=" * 70)
    print(f"{'Algorithm':<25} {'BoW':<10} {'2-gram':<10} {'3-gram':<10} {'4-gram':<10} {'TF-IDF':<10}")
    print("-" * 70)
    
    algos = [('Naive Bayes', 'nb'), ('Logistic Regression', 'lr'), ('SVM', 'svm'), ('KNN', 'knn')]
    
    for algo_name, algo_key in algos:
        bow_val = results['bow'][algo_key]
        ngram2_val = results['ngrams'][2][algo_key]
        ngram3_val = results['ngrams'][3][algo_key]
        ngram4_val = results['ngrams'][4][algo_key]
        tfidf_val = results['tfidf'][algo_key]
        
        print(f"{algo_name:<25} {bow_val:.4f}    {ngram2_val:.4f}    {ngram3_val:.4f}    {ngram4_val:.4f}    {tfidf_val:.4f}")


if __name__ == "__main__":
    main()
