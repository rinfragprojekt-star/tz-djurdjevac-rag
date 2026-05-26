from flask import Flask, render_template, request, jsonify
from rag import generate_answer
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("message", "")

    if not question:
        return jsonify({
            "answer": "Niste poslali pitanje.",
            "sources": []
        })

    try:
        answer, sources = generate_answer(question)
        return jsonify({
            "answer": answer,
            "sources": sources
        })
    except Exception as e:
        print("GREŠKA U /chat:", e)
        return jsonify({
            "answer": "Dogodila se greška u RAG sustavu.",
            "sources": []
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
