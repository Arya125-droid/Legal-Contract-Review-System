# import fitz  # PyMuPDF
import fitz

def annotate_pdf(input_pdf_path, output_pdf_path, classifier, label_colors):
    doc = fitz.open(input_pdf_path)
    for page in doc:
        blocks = page.get_text("blocks")
        for block in blocks:
            x0, y0, x1, y1, text = block[:5]
            clause = text.strip()
            if len(clause) < 30:
                continue
            pred = classifier(clause)[0][0]
            label = pred["label"].lower()
            confidence = pred["score"]
            rect = fitz.Rect(x0, y0, x1, y1)
            highlight = page.add_highlight_annot(rect)
            color = label_colors.get(label, (1, 1, 0))
            highlight.set_colors(stroke=color)
            highlight.set_info(
                title="AI Clause Classification",
                content=f"{label.upper()} ({confidence:.2f})"
            )
            highlight.update()
    doc.save(output_pdf_path, deflate=True)
    doc.close()
