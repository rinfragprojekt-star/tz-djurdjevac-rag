import os
import chromadb
from dotenv import load_dotenv
from google import genai
import google.generativeai as genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="vector_store")
collection = chroma_client.get_or_create_collection(name="tz_docs")


def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return response.embeddings[0].values


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

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text, sources
