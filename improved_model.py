import torch
import torch.nn as nn


class ImprovedModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(ImprovedModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # BiLSTM layer (bidirectional=True)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Fully connected layer
        # hidden_dim * 2 because BiLSTM has 2 directions
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len)

        x = self.embedding(x)
        # (batch_size, seq_len, embed_dim)

        lstm_out, (hidden, cell) = self.lstm(x)

        # hidden shape: (num_layers * 2, batch_size, hidden_dim)

        # Take forward + backward last hidden states
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        # Concatenate both directions
        combined = torch.cat((forward_hidden, backward_hidden), dim=1)
        # (batch_size, hidden_dim * 2)

        x = self.dropout(combined)

        out = self.fc(x)
        # (batch_size, num_classes)

        return out