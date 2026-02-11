# BBC News Classification: SPORT vs POLITICS

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive Natural Language Understanding (NLU) project implementing multiple machine learning algorithms for binary text classification of BBC news articles into SPORT and POLITICS categories.

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

This project demonstrates the implementation of various text classification algorithms from scratch (without high-level ML libraries like scikit-learn) to classify BBC news articles. The system compares different feature extraction techniques and classification algorithms to determine the most effective approach for distinguishing between sports and politics news articles.

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
| Naive Bayes       | 0.9752 | 0.9653 | 0.9455 | 0.9010 | 0.9752 | 0.9653 | 0.9752 |
| Logistic Reg      | 0.9851 | 0.9752 | 0.9554 | 0.9208 | 0.9851 | 0.9752 | 0.9851 |
| SVM               | 0.9851 | 0.9752 | 0.9554 | 0.9208 | 0.9851 | 0.9752 | 0.9851 |
| KNN (k=5)         | 0.9703 | 0.9554 | 0.9307 | 0.8911 | 0.9703 | 0.9604 | 0.9703 |

### Key Findings

- **Best Performance**: Logistic Regression and SVM with BoW/TF-IDF (~98.5% accuracy)
- **Feature Impact**: Simpler features (BoW, TF-IDF) outperform complex n-grams
- **N-gram Trends**: Performance degrades with higher n-gram orders due to sparsity
- **Consistency**: All models show similar trends across feature types

## 📖 Documentation

For detailed technical analysis, methodology, and insights, see:
- **[REPORT.md](REPORT.md)**: Comprehensive 5+ page technical report
- **[GitHub Pages](https://viveksapkal2793.github.io/NLU_Assignment_1/)**: Interactive documentation

## Authors

- **Vivek Sapkal** - [viveksapkal2793](https://github.com/viveksapkal2793)