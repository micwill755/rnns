from rnn_nlp import SimpleRNN, softmax, cross_entropy_loss
import numpy as np
import urllib.request

def download_book(url, filename):
    """Download a book from Project Gutenberg"""
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
        return True
    except:
        print(f"Failed to download {filename}")
        return False

def load_text_data():
    """Load book text - try to download, fallback to sample"""
    
    # Try to download Alice in Wonderland from Project Gutenberg
    book_url = "https://www.gutenberg.org/files/11/11-0.txt"
    filename = "alice.txt"
    
    if download_book(book_url, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"Loaded book: {len(text)} characters")
            
            # Clean up - remove Project Gutenberg header/footer
            start = text.find("CHAPTER I")
            end = text.find("THE END")
            if start > 0 and end > 0:
                text = text[start:end]
                print(f"Cleaned text: {len(text)} characters")
            
            return text
        except:
            print("Error reading downloaded file")
    
    # Fallback to larger sample text
    print("Using fallback sample text")
    return """
    Alice was beginning to get very tired of sitting by her sister on the bank, 
    and of having nothing to do: once or twice she had peeped into the book her 
    sister was reading, but it had no pictures or conversations in it, 'and what 
    is the use of a book,' thought Alice 'without pictures or conversation?'
    
    So she was considering in her own mind (as well as she could, for the hot 
    day made her feel very sleepy and stupid), whether the pleasure of making a 
    daisy-chain would be worth the trouble of getting up and picking the daisies, 
    when suddenly a White Rabbit with pink eyes ran close by her.
    """ * 50  # Repeat for more data

# Load the text
text = load_text_data()

# Prepare data
chars = sorted(list(set(text)))  # Sort for consistency
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"\nVocabulary size: {vocab_size}")
print(f"Text length: {len(text):,} characters")
print(f"Sample chars: {chars[:20]}...")

# Convert to indices
data = [char_to_idx[ch] for ch in text]

# Initialize RNN
hidden_size = 128
rnn = SimpleRNN(vocab_size, hidden_size, vocab_size)

# Training parameters
seq_length = 50
batch_size = 64
learning_rate = 1e-3
epochs = 200

print(f"\nTraining on real book text...")
print(f"Sequences per epoch: {batch_size}")
print(f"Sequence length: {seq_length}")

for epoch in range(epochs):
    total_loss = 0
    
    # Multiple sequences per epoch
    for batch in range(batch_size):
        # Random sequence from book
        max_start = len(data) - seq_length - 1
        start_idx = np.random.randint(0, max_start)
        
        inputs = data[start_idx:start_idx + seq_length]
        target = data[start_idx + seq_length]
        
        # Forward pass
        y_pred, _ = rnn.forward(inputs)
        
        # Calculate loss
        loss = cross_entropy_loss(y_pred, target)
        total_loss += loss
        
        # Backward pass
        d_y = softmax(y_pred)
        d_y[0, target] -= 1
        rnn.backward(d_y, learning_rate)
    
    avg_loss = total_loss / batch_size
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}, Loss: {avg_loss:.4f}")

print("\nTraining complete!")

# Generate text like the book
def generate_text(rnn, start_text="Alice", length=200):
    result = start_text
    
    # Convert start text to indices
    input_seq = [char_to_idx.get(ch, 0) for ch in start_text]
    
    for _ in range(length - len(start_text)):
        # Use recent context
        context = input_seq[-seq_length:] if len(input_seq) >= seq_length else input_seq
        
        y_pred, _ = rnn.forward(context)
        probs = softmax(y_pred)[0]
        
        # Sample next character
        next_idx = np.random.choice(len(probs), p=probs)
        next_char = idx_to_char[next_idx]
        
        result += next_char
        input_seq.append(next_idx)
    
    return result

# Generate book-like text
print("\nGenerated text (trained on real book):")
print("=" * 50)
sample = generate_text(rnn, "Alice was ", 300)
print(sample)
print("=" * 50)