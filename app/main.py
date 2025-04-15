from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from app.model import classifier, label_colors
from app.utils import annotate_pdf
import shutil
import tempfile
import os

app = FastAPI(title="AI-Powered Legal Clause Classifier 🚀")


@app.post("/annotate")
async def annotate_contract(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_input:
        shutil.copyfileobj(file.file, tmp_input)
        input_path = tmp_input.name

    output_path = input_path.replace(".pdf", "_annotated.pdf")
    annotate_pdf(input_path, output_path, classifier, label_colors)

    return FileResponse(output_path, filename="annotated_contract.pdf", media_type="application/pdf")

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Legal Clause Classifier API!"}

