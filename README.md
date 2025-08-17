# AI-Based Legal Contract Review System

An AI-powered Natural Language Processing (NLP) tool to automatically analyze legal contracts, identify important clauses, and highlight them in the original PDF. The system uses a fine-tuned BERT model on the [CUAD](https://github.com/TheAtticusProject/cuad) dataset and provides a web-based Streamlit interface for easy upload and clause review.

---


## Features

-  Upload and annotate legal contracts in PDF format.
-  Automatically classify clauses into legal categories:
    * Termination
    * Confidentiality
    * Dispute Resolution
    * Jurisdiction
-  Highlights clauses in the original PDF using PyMuPDF.
-  Streamlit UI for a smooth and intuitive user experience.
-  Download the annotated PDF.
-  Displays top 3 predicted clause types per block (optional).
-  Lists all extracted clauses and their predicted labels.

---


## Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/Anshuman125/Legal-Contract-Review-System.git
cd Legal-Contract-Review-System
```

### 2. Download the CUAD Dataset
To fine-tune or retrain the model, download the CUAD dataset from the official repo:

 [CUAD Dataset GitHub](https://github.com/TheAtticusProject/cuad)

Place the dataset (cuad.json, etc.) in a data/ directory.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually install the key packages:
```bash
pip install transformers streamlit fitz tqdm scikit-learn torch
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

---


### Note on Model Files
Due to GitHub's 100 MB file limit, the fine-tuned model (pytorch_model.bin or model.safetensors) is not included in this repo.

To run the app:

- Download the trained model or fine-tune it using the CUAD dataset.
- Save it to ./legal-bert-cuad/.

---


###  Model Training Info

- Fine-tuned bert-base-uncased on CUAD clause labels.
- Used HuggingFace Trainer.
- Trained for 3 epochs on MPS (Apple Silicon) — slower than CUDA.
- Evaluation: Macro F1-Score ~0.88 on unseen contract test set.

---


### Sample Prediction

```bash
Clause: "Either party may terminate this agreement with prior written notice."
🔍 Predicted: TERMINATION (0.96)
```

---


### Acknowledgements

- [CUAD Dataset – The Atticus Project](https://github.com/TheAtticusProject/cuad)
- HuggingFace Transformers
- PyMuPDF for PDF annotation

---


### License

This project is licensed under the MIT License.
