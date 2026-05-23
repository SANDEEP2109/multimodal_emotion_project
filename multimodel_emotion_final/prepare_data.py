import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = os.path.abspath("./dataset/TESS Toronto emotional speech set data")
OUTPUT_DIR = "./"

def standardize_emotion(folder_name):
    emotion = folder_name.split('_', 1)[-1].lower()
    if 'surprise' in emotion or 'surprised' in emotion:
        return 'surprise'
    return emotion

def main():
    records = []
    wav_files = glob.glob(os.path.join(DATASET_PATH, "**", "*.wav"), recursive=True)
    
    for filepath in wav_files:
        filename = os.path.basename(filepath)
        folder_name = os.path.basename(os.path.dirname(filepath))
        
        emotion = standardize_emotion(folder_name)
        
       
        parts = filename.replace('.wav', '').split('_')
        target_word = parts[1] if len(parts) >= 2 else "unknown"
        transcript = f"say the word {target_word.lower()}"
        
    
        file_id = filename.replace('.wav', '')
        
        records.append({
            "file_id": file_id,
            "filepath": filepath,
            "transcript": transcript,
            "emotion": emotion
        })
        
    df = pd.DataFrame(records)
    
 
    emotions = sorted(df['emotion'].unique())
    emotion2id = {emb: i for i, emb in enumerate(emotions)}
    df['label'] = df['emotion'].map(emotion2id)
    
  
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train_metadata.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_metadata.csv"), index=False)

    pd.DataFrame(list(emotion2id.items()), columns=['emotion', 'label']).to_csv(
        os.path.join(OUTPUT_DIR, "label_mapping.csv"), index=False
    )
    print(f"Dataset compiled! {len(train_df)} Train, {len(test_df)} Test.")

if __name__ == "__main__":
    main()