# text_pipeline/train.py
import os
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import TextEmotionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN = 6

def build_vocab(csv_path):
    df = pd.read_csv(csv_path)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for transcript in df['transcript']:
        for word in transcript.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

class TextDataset(Dataset):
    def __init__(self, csv_file, vocab):
        self.data = pd.read_csv(csv_file)
        self.vocab = vocab
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        words = row['transcript'].split()
        
        # Tokenize and pad
        tokens = [self.vocab.get(w, self.vocab["<UNK>"]) for w in words]
        if len(tokens) < MAX_SEQ_LEN:
            tokens += [self.vocab["<PAD>"]] * (MAX_SEQ_LEN - len(tokens))
        else:
            tokens = tokens[:MAX_SEQ_LEN]
            
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(row['label'], dtype=torch.long), row['file_id']

def train():
    vocab = build_vocab("../train_metadata.csv")
    torch.save(vocab, "vocab.pt")
    
    train_dataset = TextDataset("../train_metadata.csv", vocab)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = TextEmotionModel(vocab_size=len(vocab), num_classes=7).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    model.train()
    for epoch in range(10):
        epoch_loss = 0
        for tokens, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(tokens)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
    torch.save(model.state_dict(), "text_model.pth")
    
    model.eval()
    train_embeds = {}
    with torch.no_grad():
        for tokens, _, file_ids in DataLoader(train_dataset, batch_size=32):
            _, embeds = model(tokens.to(DEVICE))
            for fid, emb in zip(file_ids, embeds):
                train_embeds[fid] = emb.cpu()
    torch.save(train_embeds, "train_text_embeds.pt")
    print("Training complete. Text Vectors saved.")

if __name__ == "__main__":
    train()