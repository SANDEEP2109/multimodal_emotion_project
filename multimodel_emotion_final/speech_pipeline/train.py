import os
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import SpeechEmotionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TIME_STEPS = 100

class SpeechDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        y, sr = librosa.load(row['filepath'], sr=16000)
        y, _ = librosa.effects.trim(y)
        
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = mfcc.T # Shape: (time_steps, 40)
        
        
        if mfcc.shape[0] < MAX_TIME_STEPS:
            pad_width = MAX_TIME_STEPS - mfcc.shape[0]
            mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:MAX_TIME_STEPS, :]
            
        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(row['label'], dtype=torch.long), row['file_id']

def train():
    train_dataset = SpeechDataset("../train_metadata.csv")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = SpeechEmotionModel(num_classes=7).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(15):
        epoch_loss = 0
        for mfcc, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            mfcc, labels = mfcc.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(mfcc)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), "speech_model.pth")
    
    model.eval()
    train_embeds = {}
    with torch.no_grad():
        for mfcc, _, file_ids in DataLoader(train_dataset, batch_size=32):
            _, embeds = model(mfcc.to(DEVICE))
            for fid, emb in zip(file_ids, embeds):
                train_embeds[fid] = emb.cpu()
    torch.save(train_embeds, "train_speech_embeds.pt")
    print("Training complete. Vectors saved.")

if __name__ == "__main__":
    train()