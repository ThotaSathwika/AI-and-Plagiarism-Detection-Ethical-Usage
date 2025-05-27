# Install necessary libraries
import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from docx import Document
import PyPDF2

# Load Model and Tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4)  # 4 classes assumed
model.eval()

labels_map = {
    0: 'AI-Generated',
    1: 'Humanized-AI',
    2: 'Original',
    3: 'Plagiarized'
}

def read_file_content(file):
    text = ""
    if file.name.endswith(".txt"):
        text = file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    elif file.name.endswith(".docx"):
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        text = "Unsupported file format. Please upload .txt, .pdf, or .docx."
    return text

def predict_file(file):
    text = read_file_content(file)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = torch.max(probs).item()
    return f"Predicted: {labels_map[predicted_class]}\nConfidence: {confidence:.2f}"

# Gradio Interface
demo = gr.Interface(
    fn=predict_file,
    inputs=gr.File(file_types=[".txt", ".pdf", ".docx"]),
    outputs="text",
    title="AI & Plagiarism Content Detector",
    description="Upload a text, PDF, or DOCX file to detect whether content is AI-generated, humanized-AI, plagiarized, or original."
)

# Uncomment for local run
# demo.launch()

# Simulated Evaluation Metrics Visualization
if __name__ == "__main__":
    y_true = [0, 1, 2, 3, 0, 1, 2, 3]
    y_pred = [0, 1, 2, 3, 0, 2, 2, 3]

    report = classification_report(y_true, y_pred, target_names=list(labels_map.values()), output_dict=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics = ['precision', 'recall', 'f1-score']
    for metric in metrics:
        values = [report[label][metric] for label in labels_map.values()]
        ax.plot(list(labels_map.values()), values, marker='o', label=metric)

    ax.set_ylim(0, 1)
    ax.set_title("Precision, Recall, and F1-Score per Class")
    ax.set_ylabel("Score")
    ax.legend()
    plt.grid(True)
    plt.show()

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=list(labels_map.values())))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=list(labels_map.values()), cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()
