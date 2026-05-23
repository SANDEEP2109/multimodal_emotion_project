
import torch
import pandas as pd
from torch.utils.data import DataLoader
from model import FusionModel
from train import FusionDataset, DEVICE

def test():
    test_dataset = FusionDataset(
        "../test_metadata.csv", 
        "../speech_pipeline/test_speech_embeds.pt", 
        "../text_pipeline/test_text_embeds.pt"
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = FusionModel(num_classes=7).to(DEVICE)
    model.load_state_dict(torch.load("fusion_model.pth", weights_only=True))
    model.eval()
    
    test_fused_embeds = {}
    predictions = []
    correct, total = 0, 0
    
    with torch.no_grad():
        for s_emb, t_emb, labels, file_ids in test_loader:
            s_emb, t_emb, labels = s_emb.to(DEVICE), t_emb.to(DEVICE), labels.to(DEVICE)
            logits, fused_vecs = model(s_emb, t_emb)
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            for fid, vec in zip(file_ids, fused_vecs):
                test_fused_embeds[fid] = vec.cpu()
                
            for fid, true_lbl, pred_lbl in zip(file_ids, labels.cpu(), preds.cpu()):
                predictions.append({"file_id": fid, "true": true_lbl.item(), "fusion_pred": pred_lbl.item()})
                
    torch.save(test_fused_embeds, "test_fusion_embeds.pt")
    pd.DataFrame(predictions).to_csv("fusion_test_preds.csv", index=False)
    
    print(f"Fusion Pipeline Test Accuracy: {correct/total * 100:.2f}%")

if __name__ == "__main__":
    test()