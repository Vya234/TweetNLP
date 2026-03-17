import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    # vocab_size = total number of words
    # embed_dim = size of word vectors
    # hidden_dim = LSTM memory size
    # num_classes = 3 (sentiment i.e. negative, positive and neutral)
    def  __init__(self,vocab_size,embed_dim,hidden_dim,num_classes):
        super(BaselineModel,self).__init__()

        # converts word IDs to vectors
        self.embedding = nn.Embedding(vocab_size,embed_dim)

        # LSTM layer : reads words one by one and builds context
        self.lstm = nn.LSTM(embed_dim,hidden_dim,batch_first=True)

        # final classifier
        self.fc = nn.Linear(hidden_dim,num_classes)
    
    # forward function : defines the flow through the model
    def forward(self,x):
         # x: (batch_size, sequence_length)

        x = self.embedding(x)
        # x: (batch_size, sequence_length, embed_dim)

        lstm_out, (hidden,cell) = self.lstm(x)
        # hidden: (num_layers, batch_size, hidden_dim)

        # final hidden state, does the classification (hidden vector → sentiment scores)
        out = self.fc(hidden[-1])
        # out: (batch_size, num_classes)

        return out

