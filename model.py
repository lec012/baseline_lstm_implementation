import torch.nn as nn
import torch.nn.functional as f
import torch

class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        # LSTM layer: expects input shape (batch_size, seq_length, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(in_features = hidden_size, out_features = 3)

    def forward(self, x):
         x, _ = self.lstm(x)  # unused value represents tuple of hidden states and cell states
         x = self.linear(f.relu(x))  # relu used to mitigate vanishing gradient problem
         return x