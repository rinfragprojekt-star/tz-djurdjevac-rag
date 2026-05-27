import os
from pathlib import Path
from google import genai

DOCUMENTS_DIR = Path("documents")
MODEL_NAME = "gemini-2.0-flash"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def load_documents():
    documents = []

    if not DOCUMENTS_DIR.exists():
        return documents

    for file_path in DOCUMENTS_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8", errors="ignore")

        documents.append({
            "source": file_path.name,
            "text": text
        })

    return documents


def chunk_text(text, chunk_size=900, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def build_chunks():
    chunks = []
    documents = load_documents()

    for doc in documents:
        for index, chunk in enumerate(chunk_text(doc["text"])):
            chunks.append({
                "source": doc["source"],
                "chunk_id": index + 1,
                "text": chunk
            })

    return chunks


def score_chunk(question, chunk):
    question_words = set(question.lower().split())
    chunk_words = set(chunk.lower().split())

    return len(question_words.intersection(chunk_words))


def retrieve_context(question, top_k=4):
    chunks = build_chunks()

    scored_chunks = []
    for chunk in chunks:
        score = score_chunk(question, chunk["text"])
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    best_chunks = [
        chunk for score, chunk in scored_chunks
        if score > 0
    ][:top_k]

    return best_chunks


def generate_answer(question):
    relevant_chunks = retrieve_context(question)

    if not relevant_chunks:
        return (
            "Nisam pronašao dovoljno informacija u učitanim dokumentima za ovo pitanje.",
            []
        )

    context = ""
    sources = []

    for chunk in relevant_chunks:
        context += f"\nIZVOR: {chunk['source']} - dio {chunk['chunk_id']}\n"
        context += chunk["text"] + "\n"

        if chunk["source"] not in sources:
            sources.append(chunk["source"])

    prompt = f"""
Ti si RAG asistent za dokumente Turističke zajednice.

Odgovori isključivo na temelju konteksta ispod.
Ako odgovor nije u dokumentima, reci da informacija nije pronađena u dostupnim dokumentima.
Odgovaraj jasno, kratko i na hrvatskom jeziku.

KONTEKST:
{context}

PITANJE:
{question}

ODGOVOR:
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text, sources
