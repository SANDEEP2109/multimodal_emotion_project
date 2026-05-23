
import torch
import torch.nn as nn

class SpeechEmotionModel(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=64, num_classes=7):
        super(SpeechEmotionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, 
                            batch_first=True, bidirectional=True)
     
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
       
        lstm_out, (hn, cn) = self.lstm(x)
        
        hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1) 
        logits = self.fc(hidden)
        return logits, hidden