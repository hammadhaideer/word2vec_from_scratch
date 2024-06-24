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

# Initialize weight matrices
W1 = np.random.uniform(-0.8, 0.8, (len(vocab), embedding_dim))
W2 = np.random.uniform(-0.8, 0.8, (embedding_dim, len(vocab)))

# One-hot encoding function
def one_hot_encoding(word_idx, vocab_size):
    one_hot = np.zeros(vocab_size)
    one_hot[word_idx] = 1
    return one_hot

# Training the model
for epoch in range(epochs):
    loss = 0
    for context_word, target_word in context_target_pairs:
        context_word_vec = one_hot_encoding(context_word, len(vocab))
        
        # Forward pass
        hidden_layer = np.dot(context_word_vec, W1)
        output_layer = np.dot(hidden_layer, W2)
        
        # Softmax activation
        y_pred = np.exp(output_layer) / np.sum(np.exp(output_layer))
        
        # Calculate loss (cross-entropy)
        loss += -np.log(y_pred[target_word])
        
        # Backward pass
        y_true = one_hot_encoding(target_word, len(vocab))
        error = y_pred - y_true
        
        # Update weights
        dW2 = np.outer(hidden_layer, error)
        dW1 = np.outer(context_word_vec, np.dot(W2, error.T))
        
        W1 -= learning_rate * dW1
        W2 -= learning_rate * dW2
    
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss/len(context_target_pairs)}")

print("Training completed.")

# Get word vectors
word_vectors = {word: W1[word2idx[word]] for word in word2idx}

# Display word vectors
for word, vec in word_vectors.items():
    print(f"Word: {word}, Vector: {vec}")

# Function to find similar words
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_similar_words(word, top_n=5):
    if word not in word2idx:
        return "Word not in vocabulary"
    
    word_vec = word_vectors[word]
    similarities = {other_word: cosine_similarity(word_vec, vec) for other_word, vec in word_vectors.items() if other_word != word}
    sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_similarities[:top_n]

# Find similar words
similar_words = find_similar_words("machine")
print(f"Words similar to 'machine': {similar_words}")

