from utils import load_bbc_data, train_test_split
from bow_model import (
    train_bow_naive_bayes, predict_bow,
    train_bow_logistic_regression, predict_bow_logistic_regression,
    train_bow_svm, predict_bow_svm,
    train_bow_knn, predict_bow_knn
)
from ngram_model import train_ngram_naive_bayes, predict_ngram
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
    data = load_bbc_data("bbc-text.csv")
    train_data, test_data = train_test_split(data)

    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}\n")

    print("=" * 50)
    print("NAIVE BAYES MODELS")
    print("=" * 50)

    # Bag of Words - Naive Bayes
    bow_model = train_bow_naive_bayes(train_data)
    bow_acc = evaluate(test_data, predict_bow, bow_model)

    # N-grams - Naive Bayes
    ngram_model = train_ngram_naive_bayes(train_data, n=2)
    ngram_acc = evaluate(test_data,
                          lambda t, m: predict_ngram(t, m, 2),
                          ngram_model)

    # TF-IDF - Naive Bayes
    tfidf_model = train_tfidf_naive_bayes(train_data)
    tfidf_acc = evaluate(test_data, predict_tfidf, tfidf_model)

    print(f"Bag of Words  : {bow_acc:.4f}")
    print(f"N-grams       : {ngram_acc:.4f}")
    print(f"TF-IDF        : {tfidf_acc:.4f}")

    print("\n" + "=" * 50)
    print("LOGISTIC REGRESSION")
    print("=" * 50)

    # Logistic Regression with BoW
    lr_bow_model = train_bow_logistic_regression(train_data, epochs=50)
    lr_bow_acc = evaluate(test_data, predict_bow_logistic_regression, lr_bow_model)

    # Logistic Regression with TF-IDF
    lr_tfidf_model = train_tfidf_logistic_regression(train_data, epochs=50)
    lr_tfidf_acc = evaluate(test_data, predict_tfidf_logistic_regression, lr_tfidf_model)

    print(f"BoW           : {lr_bow_acc:.4f}")
    print(f"TF-IDF        : {lr_tfidf_acc:.4f}")

    print("\n" + "=" * 50)
    print("SVM (PERCEPTRON)")
    print("=" * 50)

    # SVM with BoW
    svm_bow_model = train_bow_svm(train_data, epochs=30)
    svm_bow_acc = evaluate(test_data, predict_bow_svm, svm_bow_model)

    # SVM with TF-IDF
    svm_tfidf_model = train_tfidf_svm(train_data, epochs=30)
    svm_tfidf_acc = evaluate(test_data, predict_tfidf_svm, svm_tfidf_model)

    print(f"BoW           : {svm_bow_acc:.4f}")
    print(f"TF-IDF        : {svm_tfidf_acc:.4f}")

    print("\n" + "=" * 50)
    print("K-NEAREST NEIGHBORS (k=5)")
    print("=" * 50)

    # KNN with BoW
    knn_bow_model = train_bow_knn(train_data)
    knn_bow_acc = evaluate(test_data,
                          lambda t, m: predict_bow_knn(t, m, k=5),
                          knn_bow_model)

    # KNN with TF-IDF
    knn_tfidf_model = train_tfidf_knn(train_data)
    knn_tfidf_acc = evaluate(test_data,
                            lambda t, m: predict_tfidf_knn(t, m, k=5),
                            knn_tfidf_model)

    print(f"BoW           : {knn_bow_acc:.4f}")
    print(f"TF-IDF        : {knn_tfidf_acc:.4f}")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"{'Algorithm':<25} {'BoW':<10} {'TF-IDF':<10}")
    print("-" * 50)
    print(f"{'Naive Bayes':<25} {bow_acc:.4f}    {tfidf_acc:.4f}")
    print(f"{'Logistic Regression':<25} {lr_bow_acc:.4f}    {lr_tfidf_acc:.4f}")
    print(f"{'SVM':<25} {svm_bow_acc:.4f}    {svm_tfidf_acc:.4f}")
    print(f"{'KNN':<25} {knn_bow_acc:.4f}    {knn_tfidf_acc:.4f}")


if __name__ == "__main__":
    main()
