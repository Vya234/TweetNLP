import torch
import torch.nn as nn
import torch.optim as optim

from utils import load_data, build_vocab, encode_text
from baseline_model import BaselineModel


# 1. Load dataset
train_texts, train_labels = load_data(
    "dataset/train_text.txt",
    "dataset/train_labels.txt"
)

val_texts, val_labels = load_data(
    "dataset/val_text.txt",
    "dataset/val_labels.txt"
)

# 2. Build vocabulary
vocab = build_vocab(train_texts)
vocab_size = len(vocab) + 1  # +1 for unknown


# 3. Encode text
train_encoded = [encode_text(t, vocab) for t in train_texts]
val_encoded = [encode_text(t, vocab) for t in val_texts]

# 4. Padding function
def pad_sequences(sequences, max_len):

    padded = []

    for seq in sequences:
        if len(seq) < max_len:
            seq = seq + [0] * (max_len - len(seq))
        else:
            seq = seq[:max_len]

        padded.append(seq)

    return padded

# 5. Apply padding
max_len = 30

train_padded = pad_sequences(train_encoded, max_len)
val_padded = pad_sequences(val_encoded, max_len)

# 6. Convert to tensors
X_train = torch.tensor(train_padded)
y_train = torch.tensor(train_labels)

X_val = torch.tensor(val_padded)
y_val = torch.tensor(val_labels)


# 7. Create model
model = BaselineModel(
    vocab_size=vocab_size,
    embed_dim=100,
    hidden_dim=128,
    num_classes=3
)

# 8. Loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 9. Training loop (BATCHED)
epochs = 5
batch_size = 64

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for i in range(0, len(X_train), batch_size):

        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 10. Evaluation
model.eval()

with torch.no_grad():

    outputs = model(X_val)

    _, predicted = torch.max(outputs, 1)

    accuracy = (predicted == y_val).sum().item() / len(y_val)

    print("Validation Accuracy:", accuracy)