import os
import chromadb
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Postavi API ključ
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="tz_docs"
)


def get_embedding(text):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text
    )
    return response["embedding"]


def search_documents(question, n_results=3):
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
        context += f"\n\nIZVOR: {meta['source']}\n{doc}"
        sources.append(meta["source"])

    return context, list(set(sources))


def generate_answer(question):
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
        model="models/gemini-2.0-flash",
        contents=prompt
    )

    return response.text, sources
