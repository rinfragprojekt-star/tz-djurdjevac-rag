import os
import re
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from google import genai

# Učitava .env lokalno, a na Cloud Runu koristi Environment Variables
load_dotenv()

DOCUMENTS_DIR = Path("documents")
MODEL_NAME = "gemini-2.0-flash"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4


STOP_WORDS = {
    "i", "u", "na", "za", "je", "su", "se", "sa", "od", "do", "po", "kao",
    "koji", "koja", "koje", "što", "kako", "gdje", "kada", "tko", "te",
    "ili", "a", "o", "iz", "pri", "to", "taj", "ta", "ovo", "ono", "mora",
    "može", "treba", "smo", "ste", "sam", "si", "bi", "biti", "ima", "imaju"
}


def get_api_key():
    """
    Dohvaća Gemini API ključ iz environment varijable.
    Lokalno može biti u .env datoteci.
    Na Cloud Runu mora biti dodan pod Variables & Secrets.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY nije postavljen. "
            "Dodaj ga lokalno u .env ili u Cloud Run Variables & Secrets."
        )

    return api_key


def get_client():
    """
    Gemini klijent se stvara tek kada korisnik pošalje pitanje.
    Tako se Flask aplikacija može normalno pokrenuti i ako API key nije odmah dostupan.
    """
    return genai.Client(api_key=get_api_key())


def normalize_text(text):
    """
    Pretvara tekst u mala slova i izvlači riječi.
    Radi i s hrvatskim znakovima.
    """
    text = text.lower()
    words = re.findall(r"[a-zA-ZčćžšđČĆŽŠĐ0-9]+", text)

    return [
        word for word in words
        if len(word) > 2 and word not in STOP_WORDS
    ]


@lru_cache(maxsize=1)
def load_documents():
    """
    Učitava sve .txt datoteke iz mape documents.
    Primjer:
    documents/djurdjevac.txt
    documents/evisitor.txt
    """
    documents = []

    if not DOCUMENTS_DIR.exists():
        print("UPOZORENJE: Mapa 'documents' ne postoji.", flush=True)
        return documents

    txt_files = list(DOCUMENTS_DIR.glob("*.txt"))

    if not txt_files:
        print("UPOZORENJE: U mapi 'documents' nema .txt datoteka.", flush=True)
        return documents

    for file_path in txt_files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore").strip()

            if not text:
                print(f"UPOZORENJE: Dokument {file_path.name} je prazan.", flush=True)
                continue

            documents.append({
                "source": file_path.name,
                "text": text
            })

        except Exception as e:
            print(f"Greška kod čitanja dokumenta {file_path.name}: {repr(e)}", flush=True)

    print("Učitani dokumenti:", [doc["source"] for doc in documents], flush=True)

    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Dijeli dokument na manje dijelove.
    Overlap znači da se dio teksta preklapa kako se ne bi izgubio kontekst.
    """
    if not text:
        return []

    if overlap >= chunk_size:
        overlap = 0

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


@lru_cache(maxsize=1)
def build_chunks():
    """
    Iz svih dokumenata izrađuje dijelove za pretraživanje.
    """
    chunks = []
    documents = load_documents()

    for doc in documents:
        doc_chunks = chunk_text(doc["text"])

        for index, chunk in enumerate(doc_chunks):
            chunks.append({
                "source": doc["source"],
                "chunk_id": index + 1,
                "text": chunk
            })

    print(f"Ukupno izrađeno chunkova: {len(chunks)}", flush=True)

    return chunks


def score_chunk(question, chunk_text_value):
    """
    Jednostavno lokalno pretraživanje.
    Gleda poklapanje riječi iz pitanja i riječi u dijelu dokumenta.
    """
    question_words = normalize_text(question)
    chunk_words = normalize_text(chunk_text_value)

    if not question_words or not chunk_words:
        return 0

    question_set = set(question_words)
    chunk_set = set(chunk_words)

    score = 0

    # Bodovi za poklapanje riječi
    common_words = question_set.intersection(chunk_set)
    score += len(common_words) * 3

    # Dodatni bodovi ako se riječ iz pitanja pojavljuje direktno u tekstu
    chunk_lower = chunk_text_value.lower()
    for word in question_set:
        if word in chunk_lower:
            score += 2

    # Dodatni bodovi ako se cijelo pitanje djelomično pojavljuje u tekstu
    question_lower = question.lower().strip()
    if question_lower and question_lower in chunk_lower:
        score += 10

    return score


def retrieve_context(question, top_k=TOP_K):
    """
    Dohvaća najrelevantnije dijelove dokumenata za korisničko pitanje.
    """
    chunks = build_chunks()

    if not chunks:
        return []

    scored_chunks = []

    for chunk in chunks:
        score = score_chunk(question, chunk["text"])
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    best_chunks = [
        chunk for score, chunk in scored_chunks
        if score > 0
    ][:top_k]

    return best_chunks


def create_prompt(question, relevant_chunks):
    """
    Sastavlja prompt koji se šalje Gemini modelu.
    """
    context_parts = []

    for chunk in relevant_chunks:
        context_parts.append(
            f"IZVOR: {chunk['source']} - dio {chunk['chunk_id']}\n"
            f"{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
Ti si RAG asistent za dokumente Turističke zajednice Đurđevac.

Tvoj zadatak:
- odgovaraj isključivo na temelju priloženog konteksta
- ako odgovor nije jasno naveden u kontekstu, reci da informacija nije pronađena u dostupnim dokumentima
- ne izmišljaj podatke
- odgovaraj na hrvatskom jeziku
- odgovor neka bude jasan, koristan i dovoljno kratak za učenike/studente
- ako je moguće, spomeni iz kojeg izvora dolazi informacija

KONTEKST:
{context}

PITANJE:
{question}

ODGOVOR:
"""

    return prompt


def generate_answer(question):
    """
    Glavna funkcija koju poziva app.py.
    Vraća:
    answer, sources
    """
    if not question or not question.strip():
        return "Niste poslali pitanje.", []

    relevant_chunks = retrieve_context(question)

    if not relevant_chunks:
        documents = load_documents()

        if not documents:
            return (
                "Nema učitanih dokumenata. Provjerite postoji li mapa 'documents' "
                "i nalazi li se u njoj datoteka poput 'djurdjevac.txt'.",
                []
            )

        return (
            "Nisam pronašao dovoljno informacija u učitanim dokumentima za ovo pitanje.",
            [doc["source"] for doc in documents]
        )

    sources = []

    for chunk in relevant_chunks:
        if chunk["source"] not in sources:
            sources.append(chunk["source"])

    prompt = create_prompt(question, relevant_chunks)

    try:
        client = get_client()

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        answer = getattr(response, "text", None)

        if not answer:
            return (
                "Model nije vratio odgovor. Provjerite Gemini API konfiguraciju.",
                sources
            )

        return answer.strip(), sources

    except Exception as e:
        print("GREŠKA U generate_answer:", repr(e), flush=True)

        return (
            f"Greška u RAG sustavu: {repr(e)}",
            sources
        )


def clear_cache():
    """
    Korisno ako promijeniš dokumente dok aplikacija radi lokalno.
    """
    load_documents.cache_clear()
    build_chunks.cache_clear()
