import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE

def evaluate_models():
    print("--- 1. Side-by-Side Accuracy / F1 Score ---")
    
    label_map = pd.read_csv("label_mapping.csv")
    class_names = label_map['emotion'].tolist()
    
    
    speech_preds = pd.read_csv("speech_pipeline/speech_test_preds.csv")
    text_preds = pd.read_csv("text_pipeline/text_test_preds.csv")
    fusion_preds = pd.read_csv("fusion_pipeline/fusion_test_preds.csv")
    
    
    df = speech_preds.merge(text_preds[['file_id', 'text_pred']], on='file_id')
    df = df.merge(fusion_preds[['file_id', 'fusion_pred']], on='file_id')
    
    y_true = df['true']
    results = []
    
    for model_name, col in [("Speech-Only", "speech_pred"), 
                            ("Text-Only", "text_pred"), 
                            ("Multimodal Fusion", "fusion_pred")]:
        y_pred = df[col]
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        results.append({"Model Variant": model_name, "Accuracy": f"{acc:.4f}", "Weighted F1": f"{f1:.4f}"})
        
       
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{model_name} Confusion Matrix")
        plt.ylabel("True Emotion")
        plt.xlabel("Predicted Emotion")
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_{model_name.replace(' ', '_')}.png")
        plt.close()
        
    metrics_df = pd.DataFrame(results)
    print(metrics_df.to_string(index=False))
    print("\nConfusion matrices saved as PNG files.\n")
    
    print("--- 3. Dimensionality Reduction (t-SNE) Separability Plots ---")

    speech_emb = torch.load("speech_pipeline/test_speech_embeds.pt", weights_only=True)
    text_emb = torch.load("text_pipeline/test_text_embeds.pt", weights_only=True)
    fusion_emb = torch.load("fusion_pipeline/test_fusion_embeds.pt", weights_only=True)
    
   
    file_ids = df['file_id'].tolist()
    labels_true = y_true.tolist()
    label_names = [class_names[i] for i in labels_true]
    
    X_speech = np.array([speech_emb[fid].numpy() for fid in file_ids])
    X_text = np.array([text_emb[fid].numpy() for fid in file_ids])
    X_fusion = np.array([fusion_emb[fid].numpy() for fid in file_ids])
    
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    embed_data = [("Speech Features", X_speech), 
                  ("Text Features", X_text), 
                  ("Joint Fusion Features", X_fusion)]
    
    for ax, (title, X) in zip(axes, embed_data):
        
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_2d = tsne.fit_transform(X)
        
        scatter = sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=label_names, 
                                  palette="tab10", ax=ax, legend='full' if ax == axes[2] else False)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout()
    plt.savefig("tsne_clusters_comparison.png")
    print("t-SNE clusters plot saved as 'tsne_clusters_comparison.png'.")

if __name__ == "__main__":
    evaluate_models()