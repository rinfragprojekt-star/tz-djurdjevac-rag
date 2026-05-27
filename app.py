from flask import Flask, render_template, request, jsonify
from rag import generate_answer
import os
import sys

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    print("DEBUG: /chat endpoint hit")
    sys.stdout.flush()

    data = request.get_json(silent=True)
    print("DEBUG: Raw JSON:", data)
    sys.stdout.flush()

    if not data:
        return jsonify({
            "answer": "Neispravan JSON format.",
            "sources": []
        }), 400

    question = data.get("message", "").strip()
    print("DEBUG: Extracted question:", question)
    sys.stdout.flush()

    if not question:
        return jsonify({
            "answer": "Niste poslali pitanje.",
            "sources": []
        })

    try:
        answer, sources = generate_answer(question)

        print("DEBUG: RAG answer generated")
        sys.stdout.flush()

        return jsonify({
            "answer": answer,
            "sources": sources
        })

    except Exception as e:
        print("GREŠKA U /chat:", e)
        sys.stdout.flush()
        return jsonify({
            "answer": "Dogodila se greška u RAG sustavu.",
            "sources": []
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
