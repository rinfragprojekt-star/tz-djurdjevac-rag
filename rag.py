import os
import re
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from google import genai

# Lokalno učitava .env datoteku
# Na Cloud Runu koristi Environment Variable
load_dotenv()

# DIREKTNO POSTAVLJEN DOKUMENT
DOCUMENT_PATH = Path("documents") / "djurdjevac.txt"

MODEL_NAME = "gemini-2.5-flash-lite"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 2


STOP_WORDS = {
    "i", "u", "na", "za", "je", "su", "se", "sa", "od", "do", "po", "kao",
    "koji", "koja", "koje", "što", "kako", "gdje", "kada", "tko", "te",
    "ili", "a", "o", "iz", "pri", "to", "taj", "ta", "ovo", "ono", "mora",
    "može", "treba", "smo", "ste", "sam", "si", "bi", "biti", "ima", "imaju",
    "će", "ne", "da", "li", "s", "pod", "nad", "prema", "kod"
}


def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY nije postavljen. "
            "Dodaj ga u .env lokalno ili u Cloud Run Variables & Secrets."
        )

    return api_key


def get_client():
    return genai.Client(api_key=get_api_key())


def normalize_text(text):
    text = text.lower()
    words = re.findall(r"[a-zA-ZčćžšđČĆŽŠĐ0-9]+", text)

    clean_words = []

    for word in words:
        if len(word) > 2 and word not in STOP_WORDS:
            clean_words.append(word)

    return clean_words


@lru_cache(maxsize=1)
def load_document():
    """
    Učitava direktno documents/djurdjevac.txt
    """
    if not DOCUMENT_PATH.exists():
        print(f"GREŠKA: Dokument ne postoji: {DOCUMENT_PATH}", flush=True)
        return None

    try:
        text = DOCUMENT_PATH.read_text(encoding="utf-8", errors="ignore").strip()

        if not text:
            print(f"GREŠKA: Dokument je prazan: {DOCUMENT_PATH}", flush=True)
            return None

        print(f"Učitan dokument: {DOCUMENT_PATH}", flush=True)

        return {
            "source": DOCUMENT_PATH.name,
            "text": text
        }

    except Exception as e:
        print(f"Greška kod čitanja dokumenta {DOCUMENT_PATH}: {repr(e)}", flush=True)
        return None


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
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
    Iz dokumenta djurdjevac.txt izrađuje dijelove za pretraživanje.
    """
    chunks = []
    document = load_document()

    if not document:
        return chunks

    doc_chunks = chunk_text(document["text"])

    for index, chunk in enumerate(doc_chunks):
        chunks.append({
            "source": document["source"],
            "chunk_id": index + 1,
            "text": chunk
        })

    print(f"Ukupno izrađeno chunkova: {len(chunks)}", flush=True)

    return chunks


def score_chunk(question, chunk_text_value):
    question_words = normalize_text(question)
    chunk_words = normalize_text(chunk_text_value)

    if not question_words or not chunk_words:
        return 0

    question_set = set(question_words)
    chunk_set = set(chunk_words)

    common_words = question_set.intersection(chunk_set)

    score = len(common_words) * 3

    chunk_lower = chunk_text_value.lower()

    for word in question_set:
        if word in chunk_lower:
            score += 2

    question_lower = question.lower().strip()

    if question_lower and question_lower in chunk_lower:
        score += 10

    return score


def retrieve_context(question, top_k=TOP_K):
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
    context_parts = []

    for chunk in relevant_chunks:
        context_parts.append(
            f"IZVOR: {chunk['source']} - dio {chunk['chunk_id']}\n"
            f"{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
Ti si RAG asistent za dokument Turističke zajednice Đurđevac.

Pravila:
- Odgovaraj isključivo na temelju priloženog konteksta.
- Ako odgovor nije jasno naveden u kontekstu, napiši da informacija nije pronađena u dokumentu.
- Nemoj izmišljati podatke.
- Odgovaraj na hrvatskom jeziku.
- Odgovor neka bude jasan, koristan i razumljiv.
- Ako je moguće, spomeni da je izvor dokument djurdjevac.txt.

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
    Prvo pokušava pronaći odgovor u documents/djurdjevac.txt.
    Ako ne pronađe relevantan dio, daje općeniti AI odgovor i jasno označava da nije iz dokumenta.
    """
    if not question or not question.strip():
        return "Niste poslali pitanje.", []

    relevant_chunks = retrieve_context(question)

    # Ako nije pronađen relevantan dio u dokumentu
    if not relevant_chunks:
        document = load_document()

        if not document:
            return (
                "Dokument nije pronađen. Provjerite postoji li datoteka "
                "'documents/djurdjevac.txt'.",
                []
            )

        try:
            client = get_client()

            prompt = f"""
Ti si AI asistent za Turističku zajednicu Đurđevac.

Za ovo pitanje nije pronađen relevantan sadržaj u dokumentu djurdjevac.txt.
Odgovori općenito, jasno i na hrvatskom jeziku.
Na početku odgovora napiši da odgovor nije pronađen u dokumentu, nego je generiran općenitim znanjem modela.

PITANJE:
{question}

ODGOVOR:
"""

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            return response.text.strip(), ["Odgovor nije pronađen u dokumentu"]

        except Exception as e:
            print("GREŠKA U općenitom AI odgovoru:", repr(e), flush=True)

            return (
                f"Greška u RAG sustavu: {repr(e)}",
                [document["source"]]
            )

    # Ako je pronađen relevantan dio u dokumentu
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

        error_text = str(e)

        if (
            "429" in error_text
            or "RESOURCE_EXHAUSTED" in error_text
            or "quota" in error_text.lower()
            or "403" in error_text
            or "PERMISSION_DENIED" in error_text
        ):
            fallback_answer = (
                "Gemini API trenutno ne može generirati odgovor, "
                "ali RAG sustav je pronašao relevantne dijelove u dokumentu.\n\n"
                "Najrelevantniji pronađeni sadržaj:\n\n"
            )

            for chunk in relevant_chunks:
                fallback_answer += f"[{chunk['source']} - dio {chunk['chunk_id']}]\n"
                fallback_answer += chunk["text"][:700] + "\n\n"

            return fallback_answer.strip(), sources

        return (
            f"Greška u RAG sustavu: {repr(e)}",
            sources
        )


def clear_cache():
    """
    Ako promijeniš djurdjevac.txt dok aplikacija radi lokalno,
    pozovi ovu funkciju ili restartaj Flask.
    """
    load_document.cache_clear()
    build_chunks.cache_clear()
