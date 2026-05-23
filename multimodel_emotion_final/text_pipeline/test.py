# text_pipeline/test.py
import torch
import pandas as pd
from torch.utils.data import DataLoader
from model import TextEmotionModel
from train import TextDataset, DEVICE

def test():
    vocab = torch.load("vocab.pt", weights_only=True)
    test_dataset = TextDataset("../test_metadata.csv", vocab)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = TextEmotionModel(vocab_size=len(vocab), num_classes=7).to(DEVICE)
    model.load_state_dict(torch.load("text_model.pth", weights_only=True))
    model.eval()
    
    test_embeds = {}
    predictions = []
    correct, total = 0, 0
    
    with torch.no_grad():
        for tokens, labels, file_ids in test_loader:
            tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
            logits, embeds = model(tokens)
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            for fid, emb in zip(file_ids, embeds):
                test_embeds[fid] = emb.cpu()
                
            for fid, true_lbl, pred_lbl in zip(file_ids, labels.cpu(), preds.cpu()):
                predictions.append({"file_id": fid, "true": true_lbl.item(), "text_pred": pred_lbl.item()})
                
    torch.save(test_embeds, "test_text_embeds.pt")
    pd.DataFrame(predictions).to_csv("text_test_preds.csv", index=False)
    
    print(f"Text Pipeline Test Accuracy: {correct/total * 100:.2f}%")

if __name__ == "__main__":
    test()