import numpy as np
import math

class EmbeddingGRU:
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        k = 1/math.sqrt(hidden_size)
        
        # Embedding layer
        self.embeddings = np.random.uniform(-0.1, 0.1, (vocab_size, embed_size))
        
        # GRU has 2 gates: reset and update
        # Reset gate
        self.Wr = np.random.uniform(-k, k, (embed_size, hidden_size))
        self.Ur = np.random.uniform(-k, k, (hidden_size, hidden_size))
        self.br = np.zeros((1, hidden_size))
        
        # Update gate
        self.Wz = np.random.uniform(-k, k, (embed_size, hidden_size))
        self.Uz = np.random.uniform(-k, k, (hidden_size, hidden_size))
        self.bz = np.zeros((1, hidden_size))
        
        # Candidate hidden state
        self.Wh = np.random.uniform(-k, k, (embed_size, hidden_size))
        self.Uh = np.random.uniform(-k, k, (hidden_size, hidden_size))
        self.bh = np.zeros((1, hidden_size))
        
        # Output layer
        self.Why = np.random.uniform(-k, k, (hidden_size, output_size))
        self.by = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
    def forward(self, inputs):
        h = np.zeros((1, self.Wr.shape[1]))
        
        self.last_inputs = inputs
        self.last_hs = {0: h}
        
        for i, word_idx in enumerate(inputs):
            # Get embedding
            x_vec = self.embeddings[word_idx:word_idx+1]
            
            # GRU GATES COMPUTATION
            # Reset gate: how much of previous hidden state to forget
            r_gate = self.sigmoid(x_vec @ self.Wr + h @ self.Ur + self.br)
            
            # Update gate: how much to update hidden state
            z_gate = self.sigmoid(x_vec @ self.Wz + h @ self.Uz + self.bz)
            
            # Candidate hidden state (uses reset gate)
            h_candidate = np.tanh(x_vec @ self.Wh + (r_gate * h) @ self.Uh + self.bh)
            
            # Final hidden state (interpolation between old and candidate)
            h = (1 - z_gate) * h + z_gate * h_candidate
            
            # Store for backprop
            self.last_hs[i + 1] = h
            
        # Final output
        y = h @ self.Why + self.by
        return y, h
    
    def backward(self, d_y, learn_rate=1e-3):
        n = len(self.last_inputs)
        
        # Initialize gradients
        gradients = {}
        for param in ['Wr', 'Ur', 'br', 'Wz', 'Uz', 'bz', 'Wh', 'Uh', 'bh', 'Why', 'by']:
            gradients[f'd{param}'] = np.zeros_like(getattr(self, param))
        
        dembeddings = np.zeros_like(self.embeddings)
        
        # Output layer gradients
        gradients['dWhy'] += self.last_hs[n].T @ d_y
        gradients['dby'] += d_y
        
        # Initialize hidden gradient
        dh = d_y @ self.Why.T
        
        # Backprop through time
        for i in reversed(range(n)):
            word_idx = self.last_inputs[i]
            x_vec = self.embeddings[word_idx:word_idx+1]
            
            h_prev = self.last_hs[i]
            h_curr = self.last_hs[i + 1]
            
            # Recompute gates
            r_gate = self.sigmoid(x_vec @ self.Wr + h_prev @ self.Ur + self.br)
            z_gate = self.sigmoid(x_vec @ self.Wz + h_prev @ self.Uz + self.bz)
            h_candidate = np.tanh(x_vec @ self.Wh + (r_gate * h_prev) @ self.Uh + self.bh)
            
            # Candidate hidden state gradients
            dh_candidate = dh * z_gate
            dh_candidate_input = dh_candidate * (1 - h_candidate**2)  # tanh derivative
            
            gradients['dWh'] += x_vec.T @ dh_candidate_input
            gradients['dUh'] += (r_gate * h_prev).T @ dh_candidate_input
            gradients['dbh'] += dh_candidate_input
            
            # Reset gate gradients
            dr_gate = dh_candidate_input @ self.Uh.T * h_prev
            dr_gate_input = dr_gate * r_gate * (1 - r_gate)  # sigmoid derivative
            
            gradients['dWr'] += x_vec.T @ dr_gate_input
            gradients['dUr'] += h_prev.T @ dr_gate_input
            gradients['dbr'] += dr_gate_input
            
            # Update gate gradients
            dz_gate = dh * (h_candidate - h_prev)
            dz_gate_input = dz_gate * z_gate * (1 - z_gate)  # sigmoid derivative
            
            gradients['dWz'] += x_vec.T @ dz_gate_input
            gradients['dUz'] += h_prev.T @ dz_gate_input
            gradients['dbz'] += dz_gate_input
            
            # Embedding gradients
            dx = (dr_gate_input @ self.Wr.T + dz_gate_input @ self.Wz.T + 
                  dh_candidate_input @ self.Wh.T)
            dembeddings[word_idx] += dx[0]
            
            # Gradients for next iteration
            if i > 0:
                dh = (dh * (1 - z_gate) + 
                      dr_gate_input @ self.Ur.T + 
                      dz_gate_input @ self.Uz.T + 
                      dh_candidate_input @ self.Uh.T * r_gate)
        
        # Update parameters
        for param in gradients:
            param_name = param[1:]
            setattr(self, param_name, getattr(self, param_name) - learn_rate * gradients[param])
        
        self.embeddings -= learn_rate * dembeddings

# Example usage
if __name__ == "__main__":
    vocab_size = 100
    embed_size = 32
    hidden_size = 16
    output_size = vocab_size
    
    # Create GRU
    gru = EmbeddingGRU(vocab_size, embed_size, hidden_size, output_size)
    
    # Test forward pass
    inputs = [5, 10, 15, 20]
    y_pred, h_final = gru.forward(inputs)
    
    print("GRU Forward Pass:")
    print(f"Input sequence: {inputs}")
    print(f"Output shape: {y_pred.shape}")
    print(f"Final hidden state shape: {h_final.shape}")
    print(f"Sample output scores: {y_pred[0, :5]}")
    
    # Test backward pass
    d_y = np.random.randn(1, vocab_size) * 0.1
    gru.backward(d_y)
    print("\nBackward pass completed successfully!")