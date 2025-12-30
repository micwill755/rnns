from rnn_nlp import SimpleRNN, softmax, cross_entropy_loss
import numpy as np

# Sample text data
text = "hello world this is a simple test for our rnn model hello world"
chars = list(set(text))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"Vocabulary: {chars}")
print(f"Vocab size: {vocab_size}")

# Convert text to indices
data = [char_to_idx[ch] for ch in text]
print(f"Data indices: {data[:10]}...")

# Initialize RNN
hidden_size = 32
rnn = SimpleRNN(vocab_size, hidden_size, vocab_size)

# Training parameters
seq_length = 10
learning_rate = 1e-2
epochs = 1000

print("\nTraining...")
for epoch in range(epochs):
    total_loss = 0
    
    # Generate random starting position
    start_idx = np.random.randint(0, len(data) - seq_length - 1)
    
    # Get sequence
    inputs = data[start_idx:start_idx + seq_length]
    target = data[start_idx + seq_length]
    
    # Forward pass
    y_pred, _ = rnn.forward(inputs)
    
    # Calculate loss
    loss = cross_entropy_loss(y_pred, target)
    total_loss += loss
    
    # Backward pass
    d_y = softmax(y_pred)
    d_y[0, target] -= 1  # Cross-entropy gradient
    rnn.backward(d_y, learning_rate)
    
    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\nTraining complete!")

# Test the model
def generate_text(rnn, start_char, length=20):
    result = start_char
    input_seq = [char_to_idx[start_char]]
    
    for _ in range(length - 1):
        y_pred, _ = rnn.forward(input_seq[-seq_length:])
        probs = softmax(y_pred)[0]
        
        # Sample from probability distribution
        next_idx = np.random.choice(len(probs), p=probs)
        next_char = idx_to_char[next_idx]
        
        result += next_char
        input_seq.append(next_idx)
    
    return result

# Generate some text
print("\nGenerated text samples:")
for i in range(3):
    sample = generate_text(rnn, 'h', 15)
    print(f"Sample {i+1}: '{sample}'")

# Test prediction accuracy
def test_accuracy(rnn, test_data, seq_len=10):
    correct = 0
    total = 0
    
    for i in range(len(test_data) - seq_len):
        inputs = test_data[i:i + seq_len]
        target = test_data[i + seq_len]
        
        y_pred, _ = rnn.forward(inputs)
        predicted = np.argmax(softmax(y_pred))
        
        if predicted == target:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0

accuracy = test_accuracy(rnn, data)
print(f"\nNext character prediction accuracy: {accuracy:.2%}")