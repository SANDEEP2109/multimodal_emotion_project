import torch
import pandas as pd
from torch.utils.data import DataLoader
from model import SpeechEmotionModel
from train import SpeechDataset, DEVICE

def test():
    test_dataset = SpeechDataset("../test_metadata.csv")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = SpeechEmotionModel(num_classes=7).to(DEVICE)
    model.load_state_dict(torch.load("speech_model.pth", weights_only=True))
    model.eval()
    
    test_embeds = {}
    predictions = []
    
    correct = 0
    total = 0
    with torch.no_grad():
        for mfcc, labels, file_ids in test_loader:
            mfcc, labels = mfcc.to(DEVICE), labels.to(DEVICE)
            logits, embeds = model(mfcc)
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            for fid, emb in zip(file_ids, embeds):
                test_embeds[fid] = emb.cpu()
                
            for fid, true_lbl, pred_lbl in zip(file_ids, labels.cpu(), preds.cpu()):
                predictions.append({"file_id": fid, "true": true_lbl.item(), "speech_pred": pred_lbl.item()})
                
    torch.save(test_embeds, "test_speech_embeds.pt")
    pd.DataFrame(predictions).to_csv("speech_test_preds.csv", index=False)
    
    print(f"Speech Pipeline Test Accuracy: {correct/total * 100:.2f}%")

if __name__ == "__main__":
    test()