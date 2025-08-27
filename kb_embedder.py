# kb_embedder.py
import chromadb
from groq import Groq
import os

client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
client_db = chromadb.Client()
collection = client_db.get_or_create_collection("personal_kb")

def embed_docs(chunks):
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[str(i)],
            documents=[chunk],
            metadatas=[{"source": f"doc_{i}"}]
        )
    print("Knowledge base embedded.")