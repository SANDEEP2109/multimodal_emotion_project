
import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, speech_dim=128, text_dim=128, num_classes=7):
        super(FusionModel, self).__init__()
     
        self.fc1 = nn.Linear(speech_dim + text_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, speech_vec, text_vec):
        fused_vec = torch.cat((speech_vec, text_vec), dim=1)
        hidden = self.relu(self.fc1(fused_vec))
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)
        return logits, fused_vec