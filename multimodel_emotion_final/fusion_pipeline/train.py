
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import FusionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FusionDataset(Dataset):
    def __init__(self, csv_file, speech_embeds_path, text_embeds_path):
        self.data = pd.read_csv(csv_file)
        self.speech_embeds = torch.load(speech_embeds_path, weights_only=True)
        self.text_embeds = torch.load(text_embeds_path, weights_only=True)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fid = row['file_id']
        label = row['label']
        s_emb = self.speech_embeds[fid]
        t_emb = self.text_embeds[fid]
        return s_emb, t_emb, torch.tensor(label, dtype=torch.long), fid

def train():
    train_dataset = FusionDataset(
        "../train_metadata.csv", 
        "../speech_pipeline/train_speech_embeds.pt", 
        "../text_pipeline/train_text_embeds.pt"
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = FusionModel(num_classes=7).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(15):
        for s_emb, t_emb, labels, _ in train_loader:
            s_emb, t_emb, labels = s_emb.to(DEVICE), t_emb.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(s_emb, t_emb)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
    torch.save(model.state_dict(), "fusion_model.pth")
    print("Multimodal Fusion Training Complete.")

if __name__ == "__main__":
    train()