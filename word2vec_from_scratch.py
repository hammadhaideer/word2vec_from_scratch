import numpy as np
import re
from collections import defaultdict

# Sample corpus
corpus = [
    "I love machine learning",
    "Machine learning is fascinating",
    "I love deep learning",
    "Deep learning is a subset of machine learning"
]

# Clean and tokenize the corpus
def preprocess_corpus(corpus):
    # Remove punctuation and convert to lowercase
    corpus = [re.sub(r"[^\w\s]", "", sentence.lower()) for sentence in corpus]
    # Split sentences into words
    tokens = [sentence.split() for sentence in corpus]
    return tokens

tokens = preprocess_corpus(corpus)

# Create a vocabulary from the tokens
vocab = set([word for sentence in tokens for word in sentence])
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

# Hyperparameters
window_size = 2
embedding_dim = 10
learning_rate = 0.01
epochs = 1000

# Generate context-target pairs
def generate_context_target_pairs(tokens, window_size):
    context_target_pairs = []
    for sentence in tokens:
        for idx, target_word in enumerate(sentence):
            # Get context words within the window size
            for w in range(-window_size, window_size + 1):
                context_pos = idx + w
                if context_pos < 0 or context_pos >= len(sentence) or w == 0:
                    continue
                context_word = sentence[context_pos]
                context_target_pairs.append((word2idx[context_word], word2idx[target_word]))
    return context_target_pairs

context_target_pairs = generate_context_target_pairs(tokens, window_size)
