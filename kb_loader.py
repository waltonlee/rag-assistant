# kb_loader.py
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_docs(folder="docs"):
    chunks = []
    for file in Path(folder).glob("*"):
        if file.suffix.lower() == ".txt":
            text = file.read_text(encoding="utf-8")
        elif file.suffix.lower() == ".pdf":
            pdf = PdfReader(str(file))
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        else:
            continue

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks.extend(splitter.split_text(text))
    return chunks