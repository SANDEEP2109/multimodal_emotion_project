Multimodal Emotion Recognition (PyTorch)

Overview
This project implements a Multimodal Emotion Recognition system using the TESS dataset.
It features three distinct, modular pipelines:
1. Speech-Only (MFCCs + BiLSTM)
2. Text-Only (Tokenization + BiLSTM)
3. Multimodal Fusion (Feature concatenation + Joint Linear Layer)

Dataset Note
To keep the submission file size reasonable, the raw TESS `.wav` audio files have been excluded. However, all extracted embeddings (`.pt`), predictions (`.csv`), and trained model weights (`.pth`) are included.
(EXcluded in GitHub, but added in Google drive)

How to Evaluate
You can view the final metrics, confusion matrices, and t-SNE plots immediately without the raw dataset by running:
`python evaluate.py`

How to Retrain from Scratch
If you wish to run the entire pipeline from scratch:
1. Download the TESS dataset online.
2. Place the unzipped folder inside a directory named `dataset/` in the root of this project.
3. Open `prepare_data.py` and ensure `DATASET_PATH` points to your folder.
4. Run the scripts in the following order:
   - `python prepare_data.py`
   - `cd speech_pipeline && python train.py && python test.py && cd ..`
   - `cd text_pipeline && python train.py && python test.py && cd ..`
   - `cd fusion_pipeline && python train.py && python test.py && cd ..`
   - `python evaluate.py`



Project Structure

project/
│
├── speech_pipeline/
├── text_pipeline/
├── fusion_pipeline/
├── outputs/
├── dataset/
├── prepare_data.py
├── evaluate.py
├── requirements.txt
└── README.md
