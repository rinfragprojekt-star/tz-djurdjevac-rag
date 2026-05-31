from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return "OK", 200

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
        from rag import generate_answer

        answer, sources = generate_answer(question)

        return jsonify({
            "answer": answer,
            "sources": sources
        })

    except Exception as e:
        import traceback

        print("GREŠKA U /chat:", repr(e), flush=True)
        traceback.print_exc()

        return jsonify({
            "answer": f"Greška u RAG sustavu: {repr(e)}",
            "sources": []
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
