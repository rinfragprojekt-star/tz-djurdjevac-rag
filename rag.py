import os
import chromadb
import google.generativeai as genai

# --- Gemini API ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

genai.configure(api_key=GEMINI_API_KEY)

# text model
model = genai.GenerativeModel("gemini-1.5-flash")

# --- ChromaDB ---
chroma_client = chromadb.PersistentClient(path="./chroma")

collection = chroma_client.get_or_create_collection(
    name="tz_djurdjevac_rag",
    metadata={"hnsw:space": "cosine"}
)

# --- Embedding preko Google Generative AI ---
def embed(text: str):
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return response["embedding"]

def generate_answer(question: str):
    q_emb = embed(question)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=3,
    )

    docs = results.get("documents", [[]])
    sources = docs[0] if docs and docs[0] else []
    context = "\n".join(sources)

    prompt = f"""
You are a helpful assistant for TZ Đurđevac.
Use the context below to answer the question.

CONTEXT:
{context}

QUESTION:
{question}
"""

    response = model.generate_content(prompt)
    answer = response.text or ""

    return answer, sources
