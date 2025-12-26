import numpy as np
import math

class EmbeddingRNN:
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        k = 1/math.sqrt(hidden_size)
        
        # Embedding layer (like transformers)
        self.embeddings = np.random.uniform(-0.1, 0.1, (vocab_size, embed_size))
        
        # RNN weights (now use embed_size instead of vocab_size)
        self.Wxh = np.random.uniform(-k, k, (embed_size, hidden_size))
        self.Whh = np.random.uniform(-k, k, (hidden_size, hidden_size))
        self.Why = np.random.uniform(-k, k, (hidden_size, output_size))
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))
        
    def forward(self, inputs):
        h = np.zeros((1, self.Whh.shape[0]))
        self.last_inputs = inputs
        self.last_hs = {0: h}
        
        for i, word_idx in enumerate(inputs):
            # Get dense embedding
            x_vec = self.embeddings[word_idx:word_idx+1]  # Shape: (1, embed_size)
            
            # Forward pass (same as before)
            h = np.tanh(x_vec @ self.Wxh + h @ self.Whh + self.bh)
            self.last_hs[i + 1] = h
            
        y = h @ self.Why + self.by
        return y, h
    
    def backward(self, d_y, learn_rate=1e-3):
        n = len(self.last_inputs)
        
        # Initialize gradients for ALL parameters (including embeddings!)
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dembeddings = np.zeros_like(self.embeddings)  # NEW: embedding gradients
        
        # Output layer gradients
        dWhy += self.last_hs[n].T @ d_y
        dby += d_y
        
        # Backprop through time
        dh = d_y @ self.Why.T
        
        for i in reversed(range(n)):
            # Tanh derivative
            temp = 1 - self.last_hs[i + 1] ** 2
            dh = dh * temp
            dbh += dh
            
            # Gradient w.r.t input (which comes from embeddings)
            dx = dh @ self.Wxh.T  # Shape: (1, embed_size)
            
            # Update embedding gradients for this specific word
            word_idx = self.last_inputs[i]
            dembeddings[word_idx] += dx[0]  # Accumulate gradient for this word
            
            # Other RNN gradients
            x_vec = self.embeddings[word_idx:word_idx+1]
            dWxh += x_vec.T @ dh
            
            if i > 0:
                dWhh += self.last_hs[i].T @ dh
                dh = dh @ self.Whh.T
        
        # Update ALL parameters (including embeddings!)
        self.Wxh -= learn_rate * dWxh
        self.Whh -= learn_rate * dWhh
        self.Why -= learn_rate * dWhy
        self.bh -= learn_rate * dbh
        self.by -= learn_rate * dby
        self.embeddings -= learn_rate * dembeddings  # Embeddings learn too!

# Example usage
vocab_size = 1000
embed_size = 128    # Much smaller than vocab_size
hidden_size = 64
output_size = vocab_size

rnn = EmbeddingRNN(vocab_size, embed_size, hidden_size, output_size)