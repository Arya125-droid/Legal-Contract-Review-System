from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("./legal-bert-cuad")
tokenizer = AutoTokenizer.from_pretrained("./legal-bert-cuad")

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=1)

label_colors = {
    "termination": (1, 1, 0),
    "confidentiality": (1, 0.8, 0.5),
    "dispute_resolution": (0.8, 1, 1),
    "jurisdiction": (0.9, 0.9, 1)
}
