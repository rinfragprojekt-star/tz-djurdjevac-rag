import os
import chromadb
from dotenv import load_dotenv
import google.generativeai as genai

# Učitaj .env datoteku (lokalno)
load_dotenv()

print("DEBUG GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)


# Inicijalizacija ChromaDB klijenta
chroma_client = chromadb.Client()

# Kreiraj ili dohvatite kolekciju
collection = chroma_client.get_or_create_collection(name="tz_docs")


def get_embedding(text: str):
    """Generira embedding za zadani tekst pomoću Gemini embedding modela."""
    response = genai.embed_content(
        model="embedding-001",
        content=text
    )

    # zaštita ako embedding ne postoji
    if "embedding" not in response or response["embedding"] is None:
        return [0.0] * 768  # fallback embedding

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

    # U novijim verzijama response.text može biti None → koristi .candidates
    if response.text:
        answer = response.text
    else:
        answer = response.candidates[0].content.parts[0].text

    return answer, sources
