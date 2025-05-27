# AI-and-Plagiarism-Detection-Ethical-Usage
 🧠 AI and Plagiarism: Detection & Ethical Usage

This project is a content classification system that distinguishes between:
- AI-Generated
- Humanized-AI
- Original
- Plagiarized content

📌 Objective
The aim is to support academic integrity by identifying content generated with AI, including humanized outputs that evade standard plagiarism detectors.

⚙️ Technology Stack
- Python
- HuggingFace Transformers (BERT)
- PyTorch
- scikit-learn
- matplotlib
- Gradio
- PyPDF2 & python-docx for file reading

💻 Installation
```bash
pip install -r requirements.txt
```

🚀 How to Run
```bash
python app.py
```

📂 Supported File Types
- .txt
- .pdf
- .docx

📊 Outputs
- Predicted label: AI-Generated / Humanized-AI / Original / Plagiarized
- Confidence score
- Visualization of precision, recall, F1-score
- Confusion matrix

📚 Features
- Advanced file parsing from various formats
- Transformer-based prediction
- Semantic similarity evaluation
- Modular architecture

🔐 License
This project is intended for academic research and demonstration purposes only.
