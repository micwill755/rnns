import numpy as np
import math

class SimpleRNN:
    def __init__(self, vocab_size, hidden_size, output_size):
        # Xavier initialization
        k = 1/math.sqrt(hidden_size)
        self.Wxh = np.random.uniform(-k, k, (vocab_size, hidden_size))  # input to hidden
        self.Whh = np.random.uniform(-k, k, (hidden_size, hidden_size))  # hidden to hidden
        self.Why = np.random.uniform(-k, k, (hidden_size, output_size))  # hidden to output
        self.bh = np.zeros((1, hidden_size))  # hidden bias
        self.by = np.zeros((1, output_size))  # output bias
        
    def forward(self, inputs):
        h = np.zeros((1, self.Whh.shape[0]))
        self.last_inputs = inputs
        self.last_hs = {0: h}
        
        for i, x in enumerate(inputs):
            # One-hot encode input
            x_vec = np.zeros((1, self.Wxh.shape[0]))
            x_vec[0, x] = 1
            
            # Forward pass
            h = np.tanh(x_vec @ self.Wxh + h @ self.Whh + self.bh)
            self.last_hs[i + 1] = h
            
        # Output layer
        y = h @ self.Why + self.by
        return y, h
    
    def backward(self, d_y, learn_rate=1e-3):
        n = len(self.last_inputs)
        
        # Initialize gradients
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        # Output layer gradients
        dWhy += self.last_hs[n].T @ d_y
        dby += d_y
        
        # Backprop through time
        dh = d_y @ self.Why.T
        
        for i in reversed(range(n)):
            # Tanh derivative
            temp = 1 - self.last_hs[i + 1] ** 2
            dh = dh * temp
            
            # Gradients
            dbh += dh
            
            # Input gradients
            x_vec = np.zeros((1, self.Wxh.shape[0]))
            x_vec[0, self.last_inputs[i]] = 1
            dWxh += x_vec.T @ dh
            
            # Hidden gradients
            if i > 0:
                dWhh += self.last_hs[i].T @ dh
                dh = dh @ self.Whh.T
        
        # Update weights
        self.Wxh -= learn_rate * dWxh
        self.Whh -= learn_rate * dWhh
        self.Why -= learn_rate * dWhy
        self.bh -= learn_rate * dbh
        self.by -= learn_rate * dby

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def cross_entropy_loss(y_pred, y_true):
    return -np.log(softmax(y_pred)[0, y_true])