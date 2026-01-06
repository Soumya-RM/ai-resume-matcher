import os

def parse_pdf(path):
    try:
        import fitz  # PyMuPDF
        text = ""
        with fitz.open(path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        print(f"[ERROR PDF] {os.path.basename(path)} → {e}")
        return ""


def parse_docx(path):
    try:
        from docx import Document
        doc = Document(path)
        return " ".join(p.text for p in doc.paragraphs)
    except Exception as e:
        print(f"[ERROR DOCX] {os.path.basename(path)} → {e}")
        return ""


def parse_all_resumes(folder_path):
    resumes = {}

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if file.lower().endswith(".pdf"):
            text = parse_pdf(file_path)
            if text.strip():
                resumes[file] = text

        elif file.lower().endswith(".docx"):
            text = parse_docx(file_path)
            if text.strip():
                resumes[file] = text

    return resumes
