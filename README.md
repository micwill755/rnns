# The Evolution of Recurrent Neural Networks (RNNs)

This repository demonstrates the evolution of RNN architectures from simple one-hot encoding to advanced gated mechanisms, implemented from scratch in NumPy. Each implementation builds upon the previous, showing how the field progressed to solve the vanishing gradient problem and improve sequence modeling.

## Architecture Evolution

### 1. Simple RNN with One-Hot Encoding
**Location**: `rnn/rnn_nlp.py`

The foundation of sequence modeling - a vanilla RNN using one-hot encoded inputs.

**Key Features**:
- Direct one-hot encoding of vocabulary
- Single hidden state with tanh activation
- Backpropagation through time (BPTT)
- Memory issues with long sequences

**Architecture**:
```
Input (one-hot) → Hidden State → Output
     ↓              ↓
   Wxh            Whh (recurrent)
```

**Limitations**:
- Sparse, high-dimensional input vectors
- Vanishing gradients in long sequences
- No selective memory mechanism

### 2. RNN with Embeddings
**Location**: `rnn/embedding_rnn.py`

A major improvement - replacing sparse one-hot vectors with dense learned embeddings.

**Key Improvements**:
- Dense embedding layer (vocab_size → embed_size)
- Learnable word representations
- Reduced computational complexity
- Better generalization

**Architecture**:
```
Input → Embedding → Hidden State → Output
         ↓            ↓
      Learned      Recurrent
   Representations  Processing
```

**Benefits**:
- Semantic relationships captured in embeddings
- Faster training with smaller matrices
- Foundation for modern NLP

### 3. Gated Recurrent Unit (GRU)
**Location**: `gru/embedding_gru.py`

Introducing selective memory with gating mechanisms to combat vanishing gradients.

**Key Innovations**:
- **Reset Gate**: Controls how much past information to forget
- **Update Gate**: Decides what new information to store
- Simplified architecture compared to LSTM
- Better gradient flow

**Architecture**:
```
Input → Embedding → [Reset Gate] → Candidate State
                 → [Update Gate] → Final Hidden State
```

**Gates**:
- Reset: `r = σ(Wr·x + Ur·h + br)`
- Update: `z = σ(Wz·x + Uz·h + bz)`
- Output: `h = (1-z)·h_prev + z·h_candidate`

### 4. Long Short-Term Memory (LSTM)
**Location**: `ltsm/embedding_lstm.py`

The most sophisticated architecture with separate cell and hidden states for long-term memory.

**Key Innovations**:
- **Forget Gate**: What to remove from cell state
- **Input Gate**: What new information to store
- **Output Gate**: What parts of cell state to output
- **Cell State**: Long-term memory separate from hidden state

**Architecture**:
```
Input → Embedding → [Forget Gate] → Cell State
                 → [Input Gate]  → (long-term memory)
                 → [Output Gate] → Hidden State
```

**Gates**:
- Forget: `f = σ(Wf·x + Uf·h + bf)`
- Input: `i = σ(Wi·x + Ui·h + bi)`
- Output: `o = σ(Wo·x + Uo·h + bo)`
- Cell: `C = f·C_prev + i·tanh(Wc·x + Uc·h + bc)`

## Training Examples

### Character-Level Language Modeling
Train on Alice in Wonderland to generate text:

```bash
# Simple RNN
cd rnn && python train_rnn.py

# Advanced training with real book data
cd rnn && python train_rnn_using_doc.py
```

### Sample Outputs
**Simple RNN**: Basic character patterns
**Embedding RNN**: Improved word-like structures  
**GRU**: Better long-term dependencies
**LSTM**: Most coherent text generation

## Learning Progression

1. **One-Hot → Embeddings**: Sparse to dense representations
2. **Vanilla → Gated**: Solving vanishing gradients
3. **GRU → LSTM**: Trading complexity for capability
4. **Manual → Automatic**: From scratch to frameworks

## Next Evolution: State Space Models

The journey doesn't end here! The next major evolution in sequence modeling has arrived with **State Space Models (SSMs)** like Mamba, which offer:

- Linear scaling with sequence length (vs quadratic for Transformers)
- Selective state mechanisms
- Hardware-efficient implementations
- Superior long-range dependencies

**Continue the evolution**: [State Space Models (SSMs)](https://github.com/micwill755/ssms)

## Repository Structure

```
rnns/
├── rnn/
│   ├── rnn_nlp.py              # Simple RNN with one-hot
│   ├── embedding_rnn.py        # RNN with embeddings
│   ├── train_rnn.py           # Basic training
│   └── train_rnn_using_doc.py # Advanced training
├── gru/
│   └── embedding_gru.py        # GRU implementation
├── ltsm/
│   └── embedding_lstm.py       # LSTM implementation
├── alice.txt                   # Training data
└── README.md                   # This file
```

## Key Takeaways

- **Embeddings** revolutionized input representation
- **Gating mechanisms** solved the vanishing gradient problem
- **LSTM** became the gold standard for sequence modeling
- Each architecture built upon previous innovations
- The evolution continues with modern architectures like Transformers and SSMs

---

*This implementation serves as an educational journey through the history of RNNs, showing how each innovation addressed specific limitations of its predecessors.*