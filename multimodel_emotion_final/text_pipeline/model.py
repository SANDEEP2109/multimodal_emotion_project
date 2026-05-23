import torch
import torch.nn as nn

class TextEmotionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=50, hidden_dim=64, num_classes=7):
        super(TextEmotionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, 
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(embedded)
        hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1) 
        logits = self.fc(hidden)
        return logits, hidden