from flask import Flask, render_template, request, jsonify
from rag import generate_answer

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    question = request.json.get("message")

    answer, sources = generate_answer(question)

    return jsonify({
        "answer": answer,
        "sources": sources
    })


if __name__ == "__main__":
    app.run(debug=True)