import math
from collections import defaultdict, Counter


def tokenize(text):
    return text.lower().split()


def analyze_dataset(data):
    """Concise dataset analysis focused on classification-relevant metrics.
    
    Displays:
    1. Class distribution (balance check)
    2. Document length statistics (words per document)
    3. Vocabulary statistics (unique words, overlap)
    4. Class-distinctive words (most indicative of each class)
    5. Separability analysis (expected classification difficulty)
    """
    print("=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)
    
    # 1. Class Distribution
    print("\n1. CLASS DISTRIBUTION")
    print("-" * 70)
    class_counts = defaultdict(int)
    for _, label in data:
        class_counts[label] += 1
    
    total_samples = len(data)
    for label, count in sorted(class_counts.items()):
        percentage = (count / total_samples) * 100
        print(f"{label:<12}: {count:>4} samples ({percentage:>5.2f}%)")
    
    # 2. Document Length Statistics
    print("\n2. DOCUMENT LENGTH STATISTICS")
    print("-" * 70)
    word_counts = {"SPORT": [], "POLITICS": []}
    
    for text, label in data:
        word_counts[label].append(len(tokenize(text)))
    
    print(f"{'Class':<12} {'Mean':<10} {'Min':<10} {'Max':<10}")
    print("-" * 70)
    for label in ["SPORT", "POLITICS"]:
        words = word_counts[label]
        avg = sum(words) / len(words)
        print(f"{label:<12} {avg:<10.1f} {min(words):<10} {max(words):<10}")
    
    # 3. Vocabulary Statistics
    print("\n3. VOCABULARY STATISTICS")
    print("-" * 70)
    
    vocab_sport = set()
    vocab_politics = set()
    word_freq_sport = Counter()
    word_freq_politics = Counter()
    
    for text, label in data:
        words = tokenize(text)
        if label == "SPORT":
            vocab_sport.update(words)
            word_freq_sport.update(words)
        else:
            vocab_politics.update(words)
            word_freq_politics.update(words)
    
    all_vocab = vocab_sport | vocab_politics
    overlap = vocab_sport & vocab_politics
    
    print(f"Total unique words    : {len(all_vocab)}")
    print(f"SPORT vocabulary      : {len(vocab_sport)}")
    print(f"POLITICS vocabulary   : {len(vocab_politics)}")
    print(f"Shared vocabulary     : {len(overlap)} ({len(overlap)/len(all_vocab)*100:.1f}%)")
    print(f"SPORT-only words      : {len(vocab_sport - vocab_politics)}")
    print(f"POLITICS-only words   : {len(vocab_politics - vocab_sport)}")
    
    # 4. Class-Distinctive Words
    print("\n4. TOP CLASS-DISTINCTIVE WORDS")
    print("-" * 70)
    
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                  'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 
                  'been', 'be', 'have', 'has', 'had', 'will', 'that', 'it'}
    
    distinctive_sport = get_distinctive_words(
        word_freq_sport, word_freq_politics, 
        class_counts["SPORT"], class_counts["POLITICS"], 
        stop_words
    )
    distinctive_politics = get_distinctive_words(
        word_freq_politics, word_freq_sport,
        class_counts["POLITICS"], class_counts["SPORT"],
        stop_words
    )
    
    print("\nSPORT distinctive words:")
    top_sport = [word for word, _ in distinctive_sport[:10]]
    print(f"  {', '.join(top_sport)}")
    
    print("\nPOLITICS distinctive words:")
    top_politics = [word for word, _ in distinctive_politics[:10]]
    print(f"  {', '.join(top_politics)}")
    
    # 5. Feature Separability Analysis
    print("\n5. CLASS SEPARABILITY ANALYSIS")
    print("-" * 70)
    
    # Calculate vocabulary overlap ratio (lower = more separable)
    overlap_ratio = len(overlap) / len(all_vocab)
    
    # Calculate distinctive word ratio (higher = more separable)
    distinctive_ratio = (len(vocab_sport - vocab_politics) + len(vocab_politics - vocab_sport)) / len(all_vocab)
    
    print(f"Vocabulary overlap ratio    : {overlap_ratio:.3f}")
    print(f"Distinctive words ratio     : {distinctive_ratio:.3f}")
    
    if distinctive_ratio > 0.5:
        separability = "HIGH - Classes have distinct vocabularies"
    elif distinctive_ratio > 0.3:
        separability = "MODERATE - Some vocabulary overlap"
    else:
        separability = "LOW - Significant vocabulary overlap"
    
    print(f"Expected separability       : {separability}")
    
    print("\n" + "=" * 70 + "\n")
    
    return {
        'total_samples': total_samples,
        'class_counts': class_counts,
        'vocab_size': len(all_vocab),
        'separability': distinctive_ratio
    }


def get_distinctive_words(freq1, freq2, count1, count2, stop_words):
    """Calculate distinctive words using frequency ratio.
    
    Identifies words that appear much more frequently in one class than another.
    Higher ratio = more distinctive.
    
    Filters:
    - Stop words (common words like 'the', 'a')
    - Short words (< 3 characters)
    - Rare words (< 5 occurrences)
    """
    distinctive = []
    
    for word in freq1:
        if word in stop_words or len(word) < 3 or freq1[word] < 5:
            continue
        
        # Normalize frequencies by class size for fair comparison
        norm_freq1 = freq1[word] / count1
        norm_freq2 = freq2.get(word, 0) / count2 if count2 > 0 else 0
        
        # Calculate distinctiveness score (ratio with smoothing)
        if norm_freq2 == 0:
            score = norm_freq1 * 10  # High score for class-exclusive words
        else:
            score = norm_freq1 / (norm_freq2 + 0.01)  # Ratio with smoothing
        
        distinctive.append((word, score))
    
    distinctive.sort(key=lambda x: x[1], reverse=True)
    return distinctive


if __name__ == "__main__":
    from utils import load_bbc_data
    data = load_bbc_data("bbc-text.csv")
    analyze_dataset(data)