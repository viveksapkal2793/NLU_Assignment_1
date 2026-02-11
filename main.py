from utils import load_bbc_data, train_test_split
from data_analysis import analyze_dataset
from bow_model import *
from ngram_model import *
from tfidf_model import *

def evaluate(test_data, predict_fn, model):
    """Evaluate model accuracy on test data."""
    correct = sum(1 for text, label in test_data if predict_fn(text, model) == label)
    return correct / len(test_data)

def train_and_evaluate(train_fn, predict_fn, train_data, test_data, **kwargs):
    """Train model and evaluate on test data in one step."""
    model = train_fn(train_data, **kwargs)
    return evaluate(test_data, predict_fn, model)

def main():
    # Load and analyze data
    data = load_bbc_data("bbc-text.csv")
    analyze_dataset(data)  # Display dataset statistics
    train_data, test_data = train_test_split(data, ratio=0.8)  # 80-20 split
    
    print("=" * 80)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 80)
    print(f"\nTraining: {len(train_data)} | Testing: {len(test_data)}\n")

    # Configuration: specify n-gram sizes to test
    ngram_sizes = [2, 3, 4]  # Bigrams, trigrams, quadgrams
    mixed_ngrams = [2, 3]  # Mixed (1+2), Mixed (1+2+3)
    
    # Results storage: organized by feature type and algorithm
    results = {
        'bow': {},  # Bag of Words results
        'ngrams': {n: {} for n in ngram_sizes},  # N-gram results
        'mixed': {n: {} for n in mixed_ngrams},  # Mixed n-gram results
        'tfidf': {}  # TF-IDF results
    }

    # ==================== NAIVE BAYES ====================
    print("=" * 80)
    print("NAIVE BAYES MODELS")
    print("=" * 80)

    results['bow']['nb'] = train_and_evaluate(train_bow_naive_bayes, predict_bow, train_data, test_data)
    print(f"BoW                  : {results['bow']['nb']:.4f}")

    for n in ngram_sizes:
        results['ngrams'][n]['nb'] = train_and_evaluate(train_ngram_naive_bayes, 
                                                         lambda t, m, n=n: predict_ngram(t, m, n), 
                                                         train_data, test_data, n=n)
        print(f"N-grams ({n})          : {results['ngrams'][n]['nb']:.4f}")
    
    for max_n in mixed_ngrams:
        results['mixed'][max_n]['nb'] = train_and_evaluate(train_ngram_naive_bayes_mixed, 
                                                            predict_ngram_mixed, 
                                                            train_data, test_data, max_n=max_n)
        print(f"Mixed (1-{max_n})         : {results['mixed'][max_n]['nb']:.4f}")

    results['tfidf']['nb'] = train_and_evaluate(train_tfidf_naive_bayes, predict_tfidf, train_data, test_data)
    print(f"TF-IDF               : {results['tfidf']['nb']:.4f}")

    # ==================== LOGISTIC REGRESSION ====================
    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION")
    print("=" * 80)

    results['bow']['lr'] = train_and_evaluate(train_bow_logistic_regression, 
                                               predict_bow_logistic_regression, 
                                               train_data, test_data, epochs=20)
    print(f"BoW                  : {results['bow']['lr']:.4f}")

    for n in ngram_sizes:
        results['ngrams'][n]['lr'] = train_and_evaluate(train_ngram_logistic_regression, 
                                                         predict_ngram_logistic_regression, 
                                                         train_data, test_data, n=n, epochs=20)
        print(f"N-grams ({n})          : {results['ngrams'][n]['lr']:.4f}")
    
    for max_n in mixed_ngrams:
        results['mixed'][max_n]['lr'] = train_and_evaluate(train_ngram_logistic_regression_mixed, 
                                                            predict_ngram_logistic_regression_mixed, 
                                                            train_data, test_data, max_n=max_n, epochs=20)
        print(f"Mixed (1-{max_n})         : {results['mixed'][max_n]['lr']:.4f}")

    results['tfidf']['lr'] = train_and_evaluate(train_tfidf_logistic_regression, 
                                                 predict_tfidf_logistic_regression, 
                                                 train_data, test_data, epochs=20)
    print(f"TF-IDF               : {results['tfidf']['lr']:.4f}")

    # ==================== SVM ====================
    print("\n" + "=" * 80)
    print("SVM (PERCEPTRON)")
    print("=" * 80)

    results['bow']['svm'] = train_and_evaluate(train_bow_svm, predict_bow_svm, 
                                                train_data, test_data, epochs=20)
    print(f"BoW                  : {results['bow']['svm']:.4f}")

    for n in ngram_sizes:
        results['ngrams'][n]['svm'] = train_and_evaluate(train_ngram_svm, predict_ngram_svm, 
                                                          train_data, test_data, n=n, epochs=20)
        print(f"N-grams ({n})          : {results['ngrams'][n]['svm']:.4f}")
    
    for max_n in mixed_ngrams:
        results['mixed'][max_n]['svm'] = train_and_evaluate(train_ngram_svm_mixed, 
                                                             predict_ngram_svm_mixed, 
                                                             train_data, test_data, max_n=max_n, epochs=20)
        print(f"Mixed (1-{max_n})         : {results['mixed'][max_n]['svm']:.4f}")

    results['tfidf']['svm'] = train_and_evaluate(train_tfidf_svm, predict_tfidf_svm, 
                                                  train_data, test_data, epochs=20)
    print(f"TF-IDF               : {results['tfidf']['svm']:.4f}")

    # ==================== KNN ====================
    print("\n" + "=" * 80)
    print("K-NEAREST NEIGHBORS (k=5)")
    print("=" * 80)

    results['bow']['knn'] = train_and_evaluate(train_bow_knn, 
                                                lambda t, m: predict_bow_knn(t, m, k=5), 
                                                train_data, test_data)
    print(f"BoW                  : {results['bow']['knn']:.4f}")

    for n in ngram_sizes:
        results['ngrams'][n]['knn'] = train_and_evaluate(train_ngram_knn, 
                                                          lambda t, m: predict_ngram_knn(t, m, k=5), 
                                                          train_data, test_data, n=n)
        print(f"N-grams ({n})          : {results['ngrams'][n]['knn']:.4f}")
    
    for max_n in mixed_ngrams:
        results['mixed'][max_n]['knn'] = train_and_evaluate(train_ngram_knn_mixed, 
                                                             lambda t, m: predict_ngram_knn_mixed(t, m, k=5), 
                                                             train_data, test_data, max_n=max_n)
        print(f"Mixed (1-{max_n})         : {results['mixed'][max_n]['knn']:.4f}")

    results['tfidf']['knn'] = train_and_evaluate(train_tfidf_knn, 
                                                  lambda t, m: predict_tfidf_knn(t, m, k=5), 
                                                  train_data, test_data)
    print(f"TF-IDF               : {results['tfidf']['knn']:.4f}")

    # ==================== SUMMARY ====================
    # Display comprehensive comparison table: 4 algorithms × 7 feature types
    print("\n" + "=" * 80)
    print("SUMMARY - COMPLETE COMPARISON")
    print("=" * 80)
    print(f"{'Algorithm':<20} {'BoW':<8} {'2-gram':<8} {'3-gram':<8} {'4-gram':<8} {'Mix1-2':<8} {'Mix1-3':<8} {'TF-IDF':<8}")
    print("-" * 80)
    
    for algo_name, algo_key in [('Naive Bayes', 'nb'), ('Logistic Reg', 'lr'), ('SVM', 'svm'), ('KNN', 'knn')]:
        print(f"{algo_name:<20} {results['bow'][algo_key]:.4f}  "
              f"{results['ngrams'][2][algo_key]:.4f}  "
              f"{results['ngrams'][3][algo_key]:.4f}  "
              f"{results['ngrams'][4][algo_key]:.4f}  "
              f"{results['mixed'][2][algo_key]:.4f}  "
              f"{results['mixed'][3][algo_key]:.4f}  "
              f"{results['tfidf'][algo_key]:.4f}")

if __name__ == "__main__":
    main()
