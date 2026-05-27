from flask import Flask, request, jsonify
from rag import generate_answer

app = Flask(__name__)

@app.route("/")
def home():
    return "RAG backend running!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"answer": "Neispravan JSON format.", "sources": []}), 400

    question = data.get("message", "").strip()

    if not question:
        return jsonify({"answer": "Niste poslali pitanje.", "sources": []})

    try:
        answer, sources = generate_answer(question)
        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        print("GREŠKA:", e)
        return jsonify({"answer": "Dogodila se greška u RAG sustavu.", "sources": []}), 500
