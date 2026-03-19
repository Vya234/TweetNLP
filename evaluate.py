import torch
from utils import load_data, build_vocab, encode_text
from baseline_model import BaselineModel


# 1. Load data
train_texts, train_labels = load_data(
    "dataset/train_text.txt",
    "dataset/train_labels.txt"
)

val_texts, val_labels = load_data(
    "dataset/val_text.txt",
    "dataset/val_labels.txt"
)


# 2. Build vocab (same as train.py)
vocab = build_vocab(train_texts)


# 3. Encode
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
val_padded = pad_sequences(val_encoded, max_len)


# 5. Convert to tensor
X_val = torch.tensor(val_padded)
y_val = torch.tensor(val_labels)


# 6. Load model (same config)
model = BaselineModel(
    vocab_size=len(vocab) + 1,
    embed_dim=100,
    hidden_dim=128,
    num_classes=3
)
model.load_state_dict(torch.load("model.pth"))
# 7. Evaluation
model.eval()

with torch.no_grad():

    outputs = model(X_val)

    _, predicted = torch.max(outputs, 1)


# 8. Compute metrics manually

num_classes = 3

precision = []
recall = []
f1 = []

for cls in range(num_classes):

    tp = ((predicted == cls) & (y_val == cls)).sum().item()
    fp = ((predicted == cls) & (y_val != cls)).sum().item()
    fn = ((predicted != cls) & (y_val == cls)).sum().item()

    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0

    precision.append(p)
    recall.append(r)
    f1.append(f)


# 9. Print results

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("Average F1:", sum(f1) / num_classes)