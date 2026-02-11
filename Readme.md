# Text Document Classification: SPORT vs POLITICS

A comprehensive repository implementing multiple machine learning algorithms for binary text classification of BBC news articles into SPORT and POLITICS categories.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository demonstrates the implementation of various text classification algorithms from scratch (without high-level ML libraries like scikit-learn) to classify BBC news articles. The system compares different feature extraction techniques and classification algorithms to determine the most effective approach for distinguishing between sports and politics news articles.

### Key Objectives

- Implement text classification algorithms from scratch
- Compare multiple feature extraction techniques (BoW, N-grams, TF-IDF)
- Evaluate performance across different classifiers
- Analyze dataset characteristics and model behavior
- Provide comprehensive documentation and reproducible results

## Features

- **Multiple Classification Algorithms**: Naive Bayes, Logistic Regression, SVM (Perceptron), K-Nearest Neighbors
- **Diverse Feature Extraction**: Bag of Words, N-grams (2-4), Mixed N-grams, TF-IDF
- **Comprehensive Analysis**: Dataset statistics, vocabulary analysis, class separability metrics
- **From-Scratch Implementation**: Core algorithms implemented without ML frameworks
- **Balanced Dataset**: Automatic class balancing for fair evaluation
- **Reproducible Results**: Fixed random seeds and structured evaluation pipeline

## Dataset

**Source**: [BBC Full Text and Category Dataset](https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category?resource=download)

The BBC dataset contains news articles from 2004-2005 across five categories. This project focuses on binary classification between:
- **SPORT**: Sports news articles
- **POLITICS**: Political news articles

### Dataset Statistics

- **Total Articles**: 1010 (balanced)
- **Training Set**: 808 articles (80%)
- **Test Set**: 202 articles (20%)
- **Average Document Length**: ~350 words
- **Vocabulary Size**: ~15,000 unique words
- **Class Distribution**: 50% SPORT, 50% POLITICS (balanced)

## Installation

### Prerequisites

- Python 3.7 or higher
- No external ML libraries required (only standard library)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/viveksapkal2793/NLU_Assignment_1.git
cd NLU_Assignment_1
```

2. **Download the dataset**:
   - Download `bbc-text.csv` from [Kaggle](https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category)
   - Place it in the root directory of the project

3. **Verify installation**:
```bash
python main.py
```

## Usage

### Basic Usage

Run the complete classification pipeline:

```bash
python main.py
```

This will:
1. Load and analyze the dataset
2. Train all models with different feature extraction methods
3. Evaluate on test data
4. Display comprehensive results

### Dataset Analysis Only

```bash
python data_analysis.py
```

### Custom Configuration

Modify parameters in `main.py`:

```python
# Change train-test split ratio
train_data, test_data = train_test_split(data, ratio=0.8)

# Adjust n-gram sizes
ngram_sizes = [2, 3, 4]  # Bigrams, trigrams, quadgrams

# Modify hyperparameters
train_and_evaluate(train_fn, predict_fn, train_data, test_data, epochs=20, learning_rate=0.1)
```

## Project Structure

```
NLU_Assignment_1/
│
├── main.py                 # Main execution pipeline
├── utils.py               # Data loading and preprocessing
├── data_analysis.py       # Dataset statistics and analysis
├── bow_model.py           # Bag of Words implementations
├── ngram_model.py         # N-gram implementations
├── tfidf_model.py         # TF-IDF implementations
├── bbc-text.csv           # BBC news dataset
├── README.md              # This file
├── REPORT.md              # Detailed technical report
└── __pycache__/           # Python cache files
```

## Models Implemented

### Classification Algorithms

1. **Naive Bayes**
   - Probabilistic classifier with Laplace smoothing
   - Best for: Fast training, interpretable probabilities

2. **Logistic Regression**
   - Linear classifier with sigmoid activation
   - Training: Stochastic Gradient Descent
   - Best for: Balanced accuracy and speed

3. **SVM (Perceptron)**
   - Linear separator with perceptron learning
   - Best for: Maximum margin classification

4. **K-Nearest Neighbors (k=5)**
   - Instance-based learning with cosine similarity
   - Best for: Capturing local patterns

### Feature Extraction Methods

1. **Bag of Words (BoW)**: Word frequency vectors
2. **N-grams**: 2-grams, 3-grams, 4-grams for context capture
3. **Mixed N-grams**: Combined 1-grams + n-grams
4. **TF-IDF**: Term frequency-inverse document frequency weighting

## Results

### Performance Summary

| Algorithm         | BoW    | 2-gram | 3-gram | 4-gram | Mix1-2 | Mix1-3 | TF-IDF |
|-------------------|--------|--------|--------|--------|--------|--------|--------|
| Naive Bayes       | 1.0000 | 1.0000 | 0.8922 | 0.5988 | 1.0000 | 1.0000 | 0.9760 |
| Logistic Reg      | 0.9880 | 0.9880 | 0.9760 | 0.9102 | 0.9940 | 0.9760 | 0.9820 |
| SVM               | 0.9880 | 0.9820 | 0.9641 | 0.9401 | 0.9940 | 1.0000 | 0.9581 |
| KNN (k=5)         | 0.8802 | 0.9521 | 0.9760 | 0.9281 | 0.8383 | 0.7964 | 0.9880 |

### Key Findings

- **Best Performance**: Naive Bayes achieves perfect accuracy (100%) on BoW, 2-gram, Mix1-2, and Mix1-3; SVM achieves 100% on Mix1-3
- **Algorithm Strengths**: KNN performs best with TF-IDF (98.8%), while Naive Bayes excels with simpler n-gram features
- **N-gram Sparsity**: Severe performance degradation with higher n-grams (Naive Bayes: 100% → 59.88% from 2-gram to 4-gram)
- **Mixed N-grams**: Combining unigrams with bigrams/trigrams recovers performance lost in pure high-order n-grams
- **Feature Effectiveness**: BoW and mixed n-grams consistently outperform pure high-order n-grams across all algorithms

## Documentation

For detailed technical analysis, methodology, and insights, see:
- **[REPORT](https://docs.google.com/document/d/1T4xzdP7oDO0iAoqYj88pV3qa9AcohpRXoZC8lP3NLmE/edit?usp=sharing)**: Comprehensive technical report
- **[GitHub Pages](https://viveksapkal2793.github.io/NLU_Assignment_1/)**: Interactive documentation

## Authors

- [Vivek Sapkal (B22AI066)](b22ai066@iitj.ac.in)