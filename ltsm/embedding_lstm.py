import numpy as np
import math

class EmbeddingLSTM:
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        k = 1/math.sqrt(hidden_size)
        
        # Embedding layer (same as RNN)
        self.embeddings = np.random.uniform(-0.1, 0.1, (vocab_size, embed_size))
        
        # LSTM has 4 gates, each needs input and hidden weights
        # Forget gate
        self.Wf = np.random.uniform(-k, k, (embed_size, hidden_size))
        self.Uf = np.random.uniform(-k, k, (hidden_size, hidden_size))
        self.bf = np.zeros((1, hidden_size))
        
        # Input gate  
        self.Wi = np.random.uniform(-k, k, (embed_size, hidden_size))
        self.Ui = np.random.uniform(-k, k, (hidden_size, hidden_size))
        self.bi = np.zeros((1, hidden_size))
        
        # Candidate values
        self.Wc = np.random.uniform(-k, k, (embed_size, hidden_size))
        self.Uc = np.random.uniform(-k, k, (hidden_size, hidden_size))
        self.bc = np.zeros((1, hidden_size))
        
        # Output gate
        self.Wo = np.random.uniform(-k, k, (embed_size, hidden_size))
        self.Uo = np.random.uniform(-k, k, (hidden_size, hidden_size))
        self.bo = np.zeros((1, hidden_size))
        
        # Final output layer (same as RNN)
        self.Why = np.random.uniform(-k, k, (hidden_size, output_size))
        self.by = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
        
    def forward(self, inputs):
        h = np.zeros((1, self.Wf.shape[1]))  # Hidden state
        c = np.zeros((1, self.Wf.shape[1]))  # Cell state (NEW!)
        
        self.last_inputs = inputs
        self.last_hs = {0: h}
        self.last_cs = {0: c}  # Store cell states too
        
        for i, word_idx in enumerate(inputs):
            # Get embedding
            x_vec = self.embeddings[word_idx:word_idx+1]
            
            # LSTM GATES COMPUTATION
            # Forget gate: what to forget from previous cell state
            f_gate = self.sigmoid(x_vec @ self.Wf + h @ self.Uf + self.bf)
            
            # Input gate: what new information to store
            i_gate = self.sigmoid(x_vec @ self.Wi + h @ self.Ui + self.bi)
            
            # Candidate values: new information that could be stored
            c_candidate = np.tanh(x_vec @ self.Wc + h @ self.Uc + self.bc)
            
            # Output gate: what parts of cell state to output
            o_gate = self.sigmoid(x_vec @ self.Wo + h @ self.Uo + self.bo)
            
            # UPDATE CELL STATE (key difference from RNN!)
            c = f_gate * c + i_gate * c_candidate
            
            # UPDATE HIDDEN STATE
            h = o_gate * np.tanh(c)
            
            # Store for backprop
            self.last_hs[i + 1] = h
            self.last_cs[i + 1] = c
            
        # Final output (same as RNN)
        y = h @ self.Why + self.by
        return y, h
    
    def backward(self, d_y, learn_rate=1e-3):
        n = len(self.last_inputs)
        
        # Initialize gradients for all parameters
        gradients = {}
        for param in ['Wf', 'Uf', 'bf', 'Wi', 'Ui', 'bi', 
                     'Wc', 'Uc', 'bc', 'Wo', 'Uo', 'bo', 'Why', 'by']:
            gradients[f'd{param}'] = np.zeros_like(getattr(self, param))
        
        dembeddings = np.zeros_like(self.embeddings)
        
        # Output layer gradients
        gradients['dWhy'] += self.last_hs[n].T @ d_y
        gradients['dby'] += d_y
        
        # Initialize hidden and cell gradients
        dh = d_y @ self.Why.T
        dc = np.zeros_like(dh)
        
        # Backprop through time
        for i in reversed(range(n)):
            word_idx = self.last_inputs[i]
            x_vec = self.embeddings[word_idx:word_idx+1]
            
            h_prev = self.last_hs[i]
            c_prev = self.last_cs[i]
            h_curr = self.last_hs[i + 1]
            c_curr = self.last_cs[i + 1]
            
            # Recompute gates (needed for gradients)
            f_gate = self.sigmoid(x_vec @ self.Wf + h_prev @ self.Uf + self.bf)
            i_gate = self.sigmoid(x_vec @ self.Wi + h_prev @ self.Ui + self.bi)
            c_candidate = np.tanh(x_vec @ self.Wc + h_prev @ self.Uc + self.bc)
            o_gate = self.sigmoid(x_vec @ self.Wo + h_prev @ self.Uo + self.bo)
            
            # Output gate gradients
            do_gate = dh * np.tanh(c_curr)
            do_gate_input = do_gate * o_gate * (1 - o_gate)  # sigmoid derivative
            
            gradients['dWo'] += x_vec.T @ do_gate_input
            gradients['dUo'] += h_prev.T @ do_gate_input
            gradients['dbo'] += do_gate_input
            
            # Cell state gradients
            dc += dh * o_gate * (1 - np.tanh(c_curr)**2)  # tanh derivative
            
            # Forget gate gradients
            df_gate = dc * c_prev
            df_gate_input = df_gate * f_gate * (1 - f_gate)
            
            gradients['dWf'] += x_vec.T @ df_gate_input
            gradients['dUf'] += h_prev.T @ df_gate_input
            gradients['dbf'] += df_gate_input
            
            # Input gate gradients
            di_gate = dc * c_candidate
            di_gate_input = di_gate * i_gate * (1 - i_gate)
            
            gradients['dWi'] += x_vec.T @ di_gate_input
            gradients['dUi'] += h_prev.T @ di_gate_input
            gradients['dbi'] += di_gate_input
            
            # Candidate gradients
            dc_candidate = dc * i_gate
            dc_candidate_input = dc_candidate * (1 - c_candidate**2)  # tanh derivative
            
            gradients['dWc'] += x_vec.T @ dc_candidate_input
            gradients['dUc'] += h_prev.T @ dc_candidate_input
            gradients['dbc'] += dc_candidate_input
            
            # Embedding gradients
            dx = (do_gate_input @ self.Wo.T + df_gate_input @ self.Wf.T + 
                  di_gate_input @ self.Wi.T + dc_candidate_input @ self.Wc.T)
            dembeddings[word_idx] += dx[0]
            
            # Gradients for next iteration
            if i > 0:
                dh = (do_gate_input @ self.Uo.T + df_gate_input @ self.Uf.T + 
                      di_gate_input @ self.Ui.T + dc_candidate_input @ self.Uc.T)
                dc = dc * f_gate
        
        # Update all parameters
        for param in gradients:
            param_name = param[1:]  # Remove 'd' prefix
            setattr(self, param_name, getattr(self, param_name) - learn_rate * gradients[param])
        
        self.embeddings -= learn_rate * dembeddings

# Example usage and comparison
if __name__ == "__main__":
    vocab_size = 100
    embed_size = 32
    hidden_size = 16
    output_size = vocab_size
    
    # Create LSTM
    lstm = EmbeddingLSTM(vocab_size, embed_size, hidden_size, output_size)
    
    # Test forward pass
    inputs = [5, 10, 15, 20]
    y_pred, h_final = lstm.forward(inputs)
    
    print("LSTM Forward Pass:")
    print(f"Input sequence: {inputs}")
    print(f"Output shape: {y_pred.shape}")
    print(f"Final hidden state shape: {h_final.shape}")
    print(f"Sample output scores: {y_pred[0, :5]}")
    
    # Test backward pass
    d_y = np.random.randn(1, vocab_size) * 0.1
    lstm.backward(d_y)
    print("\nBackward pass completed successfully!")