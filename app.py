import streamlit as st
import fitz  # PyMuPDF
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import tempfile
import pandas as pd
import requests

# Load model and tokenizer once for local processing (Streamlit's direct processing)
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("./legal-bert-cuad")
    tokenizer = AutoTokenizer.from_pretrained("./legal-bert-cuad")
    return pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=3)

classifier = load_model()

label_colors = {
    "termination": (1, 1, 0),
    "confidentiality": (1, 0.8, 0.5),
    "dispute_resolution": (0.8, 1, 1),
    "jurisdiction": (0.9, 0.9, 1)
}

# Function for local PDF annotation (using Streamlit backend directly)
def annotate_pdf(pdf_bytes):
    clause_predictions = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks")
            for block in blocks:
                x0, y0, x1, y1, text = block[:5]
                clause = text.strip()
                if len(clause) < 30:
                    continue
                preds = classifier(clause)[0]  # top_k=3 list
                top_pred = preds[0]
                label = top_pred["label"].lower()
                confidence = top_pred["score"]

                # Save clause prediction info
                clause_predictions.append({
                    "Page": page_num,
                    "Clause": clause,
                    "Label": label.upper(),
                    "Confidence": f"{confidence:.2f}"
                })

                rect = fitz.Rect(x0, y0, x1, y1)
                highlight = page.add_highlight_annot(rect)
                color = label_colors.get(label, (1, 1, 0))
                highlight.set_colors(stroke=color)
                highlight.set_info(
                    title="AI Clause Classification",
                    content=f"{label.upper()} ({confidence:.2f})"
                )
                highlight.update()

        output_bytes = doc.write()
        return output_bytes, clause_predictions

# Streamlit UI
st.title("AI-Powered Contract Clause Annotator")

uploaded_file = st.file_uploader("Upload your contract (PDF)", type="pdf")

if uploaded_file:
    # Option to use local processing (Streamlit-only) or external API
    use_fastapi = st.radio("Choose how to process the file:", ("Local Processing", "FastAPI Backend"))

    if use_fastapi == "Local Processing":
        st.info("Reading and analyzing clauses...")
        annotated_pdf, clause_data = annotate_pdf(uploaded_file.read())
        st.success("PDF annotated successfully!")

        # Show table of predictions
        st.subheader("Clause Predictions")
        df = pd.DataFrame(clause_data)
        st.dataframe(df, use_container_width=True)

        # Download CSV
        st.download_button(
            label="Download Predictions CSV",
            data=df.to_csv(index=False).encode(),
            file_name="clause_predictions.csv",
            mime="text/csv"
        )

        # Download PDF
        st.download_button(
            label="Download Highlighted PDF",
            data=annotated_pdf,
            file_name="annotated_contract.pdf",
            mime="application/pdf"
        )

    elif use_fastapi == "FastAPI Backend":
        with st.spinner("Sending file to FastAPI backend for annotation..."):
            # Send the file to the FastAPI backend
            files = {"file": uploaded_file.getvalue()}
            response = requests.post("http://127.0.0.1:8000/annotate", files={"file": ("contract.pdf", uploaded_file.getvalue(), "application/pdf")})

            if response.status_code == 200:
                st.success("Annotation complete!")
                st.download_button(
                    label="📥 Download Annotated PDF",
                    data=response.content,
                    file_name="annotated_contract.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("Annotation failed. Please try again.")
