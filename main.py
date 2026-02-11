from utils import load_bbc_data, train_test_split
from bow_model import train_bow_naive_bayes, predict_bow
from ngram_model import train_ngram_naive_bayes, predict_ngram
from tfidf_model import train_tfidf_naive_bayes, predict_tfidf


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

    # Bag of Words
    bow_model = train_bow_naive_bayes(train_data)
    bow_acc = evaluate(test_data, predict_bow, bow_model)

    # N-grams
    ngram_model = train_ngram_naive_bayes(train_data, n=2)
    ngram_acc = evaluate(test_data,
                          lambda t, m: predict_ngram(t, m, 2),
                          ngram_model)

    # TF-IDF
    tfidf_model = train_tfidf_naive_bayes(train_data)
    tfidf_acc = evaluate(test_data, predict_tfidf, tfidf_model)

    print("Accuracy Comparison:")
    print(f"Bag of Words  : {bow_acc:.2f}")
    print(f"N-grams      : {ngram_acc:.2f}")
    print(f"TF-IDF       : {tfidf_acc:.2f}")


if __name__ == "__main__":
    main()
