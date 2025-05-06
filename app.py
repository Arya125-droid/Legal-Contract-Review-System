import streamlit as st
import fitz  # PyMuPDF
import os
from docx import Document
import tempfile
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import requests
import json
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Contract Clause Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("📄 AI-Powered Contract Clause Analyzer")
st.markdown("Analyze and highlight important clauses in your legal contracts")

# Load model and tokenizer with cache
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("./legal-bert-cuad")
    tokenizer = AutoTokenizer.from_pretrained("./legal-bert-cuad")
    return pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=3)

# Extract text from DOCX files
def extract_text_from_docx(docx_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(docx_bytes)
        tmp_path = tmp.name
    doc = Document(tmp_path)  # Use Document directly
    os.unlink(tmp_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

# Load classifier
classifier = load_model()

# Define highlighting colors
label_colors = {
    "termination": (1, 1, 0),
    "confidentiality": (1, 0.8, 0.5),
    "dispute_resolution": (0.8, 1, 1),
    "jurisdiction": (0.9, 0.9, 1)
}

# Function for PDF annotation
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
                preds = classifier(clause)[0]
                top_pred = preds[0]
                label = top_pred["label"].lower()
                confidence = top_pred["score"]

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

# Function to analyze DOCX content
def analyze_docx(docx_bytes):
    text = extract_text_from_docx(docx_bytes)
    paragraphs = text.split('\n')
    clause_predictions = []
    
    for i, para in enumerate(paragraphs):
        if len(para.strip()) < 30:
            continue
        
        preds = classifier(para)[0]
        top_pred = preds[0]
        label = top_pred["label"].lower()
        confidence = top_pred["score"]
        
        clause_predictions.append({
            "Paragraph": i + 1,
            "Clause": para,
            "Label": label.upper(),
            "Confidence": f"{confidence:.2f}"
        })
    
    return clause_predictions

# Sidebar with DMS integration and color legend
with st.sidebar:
    st.header("Document Source")
    
    source_option = st.radio(
        "Select document source:",
        ("Upload File", "Connect to DMS")
    )
    
    if source_option == "Connect to DMS":
        dms_type = st.selectbox(
            "Select DMS type:",
            ["SharePoint", "Google Drive", "Dropbox", "OneDrive"]
        )
        
        # DMS credentials input
        if dms_type:
            st.write(f"Connect to {dms_type}")
            
            with st.expander("Connection Settings"):
                if dms_type == "SharePoint":
                    st.text_input("SharePoint URL")
                    st.text_input("Site Name")
                elif dms_type == "Google Drive":
                    st.info("Click to authenticate with Google")
                    st.button("Connect to Google Drive")
                elif dms_type == "Dropbox":
                    st.button("Connect to Dropbox")
                elif dms_type == "OneDrive":
                    st.text_input("Microsoft Account Email")
            
            # Simulated file browser for DMS
            st.write("DMS Files (Demo)")
            dms_files = ["Contract_2023.pdf", "Agreement_v2.docx", "NDA_template.pdf"]
            selected_dms_file = st.selectbox("Select a file", dms_files)
            fetch_btn = st.button("Fetch from DMS")
            
            if fetch_btn:
                st.session_state.dms_file_name = selected_dms_file
                st.session_state.using_dms = True
                st.info(f"Fetched: {selected_dms_file} (simulation)")
    
    # Color legend
    st.header("Clause Types")
    
    for label, color in label_colors.items():
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(color[0]*255), int(color[1]*255), int(color[2]*255))
        st.markdown(
            f"<div style='background-color:{hex_color};padding:10px;border-radius:5px;margin-bottom:10px;'>"
            f"<span style='color:black;font-weight:bold'>{label.upper()}</span></div>", 
            unsafe_allow_html=True
        )

# Main content area
if source_option == "Upload File" or not "using_dms" in st.session_state:
    # File uploader for multiple formats
    uploaded_file = st.file_uploader("Upload your contract document", 
                                    type=["pdf", "docx"],
                                    help="Supported formats: PDF, DOCX (Word)")
else:
    # Display DMS file info
    st.write(f"Processing DMS file: {st.session_state.dms_file_name}")
    uploaded_file = st.session_state.dms_file_name  # In a real app, this would be the file content

# Process uploaded file
if uploaded_file:
    if isinstance(uploaded_file, str):  # This is a DMS file name
        st.info(f"Analyzing {uploaded_file} from {dms_type}...")
        # In a real app, you would fetch the file from DMS here
        # For demo purposes, we'll simulate a PDF file
        st.info("This is a simulation - in a real app, the file would be fetched from your DMS")
        
        # Simulate processing results
        st.success("Analysis complete!")
        
        # Create simulated data
        clause_data = [
            {"Page": 1, "Clause": "This agreement shall terminate upon 30 days notice.", "Label": "TERMINATION", "Confidence": "0.92"},
            {"Page": 2, "Clause": "All information shall be kept confidential for a period of 5 years.", "Label": "CONFIDENTIALITY", "Confidence": "0.87"}
        ]
        
        df = pd.DataFrame(clause_data)
        st.subheader("Analysis Results")
        st.dataframe(df, use_container_width=True)
        
    else:  # This is a real uploaded file
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        use_fastapi = st.radio("Choose processing method:", ("Local Processing", "FastAPI Backend"))
        
        if use_fastapi == "Local Processing":
            with st.spinner("Analyzing document..."):
                try:
                    if file_extension == ".pdf":
                        # Process PDF
                        pdf_bytes = uploaded_file.read()
                        annotated_pdf, clause_data = annotate_pdf(pdf_bytes)
                        
                        st.success("PDF analyzed successfully!")
                        
                        # Results and download options
                        st.subheader("Analysis Results")
                        df = pd.DataFrame(clause_data)
                        st.dataframe(df, use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="Download Predictions CSV",
                                data=df.to_csv(index=False).encode(),
                                file_name="clause_predictions.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            st.download_button(
                                label="Download Highlighted PDF",
                                data=annotated_pdf,
                                file_name="annotated_contract.pdf",
                                mime="application/pdf"
                            )
                            
                    elif file_extension == ".docx":
                        # Process DOCX
                        docx_bytes = uploaded_file.read()
                        clause_data = analyze_docx(docx_bytes)
                        
                        st.success("DOCX analyzed successfully!")
                        
                        # Results and download options
                        st.subheader("Analysis Results")
                        df = pd.DataFrame(clause_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Export results
                        st.download_button(
                            label="Download Predictions CSV",
                            data=df.to_csv(index=False).encode(),
                            file_name="clause_predictions.csv",
                            mime="text/csv"
                        )
                        
                        st.info("Note: DOCX highlighting is not available. Results are provided as CSV only.")
                        
                    else:
                        st.error("Unsupported file format. Please upload a PDF or DOCX file.")
                
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
        
        elif use_fastapi == "FastAPI Backend":
            with st.spinner("Sending to FastAPI backend..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), 
                                    "application/pdf" if file_extension == ".pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
                    response = requests.post("http://127.0.0.1:8000/annotate", files=files)
                    
                    if response.status_code == 200:
                        st.success("Analysis complete!")
                        
                        # For PDF files, offer download
                        if file_extension == ".pdf":
                            st.download_button(
                                label="Download Annotated Document",
                                data=response.content,
                                file_name=f"annotated_{uploaded_file.name}",
                                mime="application/pdf"
                            )
                        else:
                            # For DOCX, display results from response
                            results = response.json()
                            st.dataframe(pd.DataFrame(results), use_container_width=True)
                    else:
                        st.error(f"Backend processing failed (Status: {response.status_code})")
                except Exception as e:
                    st.error(f"Backend connection error: {str(e)}")
                    st.info("Make sure your FastAPI backend is running at http://127.0.0.1:8000")

else:
    # Show example when no file is uploaded
    st.info("👆 Please upload a contract document to begin the analysis")
    
    # Sample results preview
    with st.expander("See example results"):
        st.image("Example.png", 
                 caption="Example of a highlighted contract")
        st.write("The system will identify and highlight different types of clauses in your contract.")

# DMS Save/Export option
if "using_dms" in st.session_state and st.session_state.using_dms:
    st.subheader("Save Results to DMS")
    save_name = st.text_input("Save as", f"Analyzed_{st.session_state.dms_file_name}")
    save_location = st.selectbox("Save to folder", ["Contracts", "Legal Review", "Shared Documents"])
    
    if st.button("Save to DMS"):
        st.success(f"Results saved to {dms_type}: {save_location}/{save_name} (simulation)")

