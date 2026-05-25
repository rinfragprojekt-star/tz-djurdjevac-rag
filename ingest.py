import os
from pypdf import PdfReader
from rag import collection, get_embedding


def read_txt(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def read_pdf(path):
    reader = PdfReader(path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def split_text(text, chunk_size=1000, overlap=150):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def ingest_documents():

    documents_folder = "documents"

    doc_id = 0

    for filename in os.listdir(documents_folder):

        path = os.path.join(documents_folder, filename)

        if filename.endswith(".txt"):
            print(f"Učitavam TXT: {filename}")

            text = read_txt(path)

        elif filename.endswith(".pdf"):
            print(f"Učitavam PDF: {filename}")

            text = read_pdf(path)

        else:
            continue

        chunks = split_text(text)

        print(f"Broj chunkova: {len(chunks)}")

        for chunk in chunks:

            embedding = get_embedding(chunk)

            collection.add(
                ids=[f"doc_{doc_id}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": filename}]
            )

            doc_id += 1

    print("Dokumenti uspješno spremljeni.")


if __name__ == "__main__":
    ingest_documents()