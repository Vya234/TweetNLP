import torch
import torch.nn as nn
import torch.optim as optim
import os

from utils import load_data, build_vocab, encode_text
from baseline_model import BaselineModel
from improved_model import ImprovedModel


# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# PATH HANDLING
if os.path.exists("/kaggle/input"):
    BASE_PATH = "/kaggle/input/datasets/vya0651/tweetnlp-project-kavya/TweetNLP/dataset"
else:
    BASE_PATH = "dataset"


# 1. Load data
train_texts, train_labels = load_data(
    os.path.join(BASE_PATH, "train_text.txt"),
    os.path.join(BASE_PATH, "train_labels.txt")
)

val_texts, val_labels = load_data(
    os.path.join(BASE_PATH, "val_text.txt"),
    os.path.join(BASE_PATH, "val_labels.txt")
)


# 2. Build vocab
vocab = build_vocab(train_texts)
vocab_size = len(vocab) + 1


# 3. Encode
train_encoded = [encode_text(t, vocab) for t in train_texts]
val_encoded = [encode_text(t, vocab) for t in val_texts]


# 4. Padding
def pad_sequences(sequences, max_len):
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            seq = seq + [0] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        padded.append(seq)
    return padded


max_len = 30
train_padded = pad_sequences(train_encoded, max_len)
val_padded = pad_sequences(val_encoded, max_len)


# 5. Tensor
X_train = torch.tensor(train_padded).to(device)
y_train = torch.tensor(train_labels).to(device)

X_val = torch.tensor(val_padded).to(device)
y_val = torch.tensor(val_labels).to(device)


# CHOOSE MODEL
MODEL_TYPE = "improved"   # "baseline" or "improved"

if MODEL_TYPE == "baseline":
    model = BaselineModel(vocab_size, 100, 128, 3)
else:
    model = ImprovedModel(vocab_size, 100, 128, 3)

model = model.to(device)


# 6. Loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 7. Training
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


# 8. Validation
model.eval()

with torch.no_grad():
    outputs = model(X_val)
    _, predicted = torch.max(outputs, 1)

    accuracy = (predicted == y_val).sum().item() / len(y_val)
    print("Validation Accuracy:", accuracy)


# 9. Save model
if os.path.exists("/kaggle/input"):
    save_path = "/kaggle/working/model.pth"
else:
    save_path = "model.pth"

torch.save(model.state_dict(), save_path)
print("Model saved at:", save_path)