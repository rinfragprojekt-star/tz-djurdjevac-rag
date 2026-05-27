import os
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import google.generativeai as genai

# Učitaj .env (lokalno)
load_dotenv()

# Konfiguracija Gemini API-ja
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"

# ChromaDB 0.4.x – in-memory (radi u Cloud Runu)
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+memory",
    persist_directory=":memory:"
))

collection = chroma_client.get_or_create_collection(name="tz_docs")


def get_embedding(text: str):
    """Generira embedding za zadani tekst pomoću Gemini embedding modela."""
    response = genai.embed_content(
        model="text-embedding-004",
        content=text
    )

    if "embedding" not in response or response["embedding"] is None:
        return [0.0] * 768

    return response["embedding"]


def search_documents(question: str, n_results: int = 3):
    """Pretražuje dokumente u ChromaDB koristeći embedding upit."""
    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = ""
    sources = []

    for doc, meta in zip(documents, metadatas):
        context += f"\n\nIZVOR: {meta.get('source', 'Nepoznato')}\n{doc}"
        sources.append(meta.get("source", "Nepoznato"))

    return context, list(set(sources))


def generate_answer(question: str):
    """Generira odgovor koristeći Gemini model i kontekst iz dokumenata."""
    context, sources = search_documents(question)

    prompt = f"""
Ti si chatbot Turističke zajednice grada Đurđevca.

Odgovaraj SAMO koristeći informacije iz konteksta.
Ako odgovor nije u dokumentima, reci:
"Nemam tu informaciju u dostupnim dokumentima."

KONTEKST:
{context}

PITANJE:
{question}
"""

    response = genai.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    if response.text:
        answer = response.text
    else:
        answer = response.candidates[0].content.parts[0].text

    return answer, sources
