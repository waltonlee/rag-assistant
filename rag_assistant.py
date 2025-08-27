from kb_loader import load_docs
from kb_embedder import embed_docs
from groq import Groq
import os
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# Load & embed (run once, or cache embeddings)
chunks = load_docs()
embed_docs(chunks)

# Initialize Groq & ChromaDB
client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
client_db = chromadb.Client()
collection = client_db.get_collection("personal_kb")

# Query function
def query_kb(query, top_k=3):
    embedding_function = DefaultEmbeddingFunction()
    query_emb = embedding_function([query])
    results = collection.query(query_embeddings=query_emb, n_results=top_k)
    docs = results["documents"][0]
    return "\n".join(docs)

# RAG function
history = []
def ask_rag(query):
    context = query_kb(query)
    system_prompt = f"Answer using only the context:\n{context}"
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": query}]
    response = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return response.choices[0].message.content

# Chat loop
console = Console()
console.print(Panel("🤖 Groq RAG Assistant — type 'quit' to exit"))

while True:
    user_input = Prompt.ask("[bold cyan]You[/]")
    if user_input.lower() in {"quit", "exit"}:
        console.print("Goodbye!")
        break
    answer = ask_rag(user_input)
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": answer})
    console.print(Panel(answer, title="Assistant"))