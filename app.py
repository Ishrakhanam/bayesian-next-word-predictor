from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from predictor import BayesianNextWordModel
from train_model import evaluate_corpus, load_corpus

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "bayesian_model.pkl"
CORPUS_PATH = BASE_DIR / "data" / "training_corpus.txt"

app = Flask(__name__, static_folder="static", template_folder="templates")


def ensure_corpus_exists() -> None:
    if not CORPUS_PATH.exists():
        corpus_text = "\n".join(load_corpus())
        CORPUS_PATH.write_text(corpus_text + "\n", encoding="utf-8")


def build_model() -> BayesianNextWordModel:
    ensure_corpus_exists()
    if MODEL_PATH.exists():
        loaded_model = BayesianNextWordModel.load_model(MODEL_PATH)
        if getattr(loaded_model, "model_version", None) == BayesianNextWordModel.MODEL_VERSION:
            return loaded_model

    model = BayesianNextWordModel(max_context=3, alpha=0.8)
    model.train(load_corpus())
    model.save_model(MODEL_PATH)
    return model


current_model = build_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "vocabulary_size": len(current_model.vocab),
        "total_tokens": current_model.total_tokens,
    })


@app.route("/api/model-info")
def model_info():
    return jsonify(current_model.metadata())


@app.route("/api/corpus")
def corpus_data():
    sentences = load_corpus()
    return jsonify({
        "source": "data/training_corpus.txt",
        "sentence_count": len(sentences),
        "sentences": sentences,
    })


@app.route("/api/metrics")
def evaluation_metrics():
    return jsonify(evaluate_corpus(load_corpus()))


@app.route("/api/predict", methods=["POST"])
def predict_next_word():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "A JSON object is required"}), 400

    text = payload.get("text", "")
    if not isinstance(text, str):
        return jsonify({"error": "The text field must be a string"}), 400

    requested_top_k = payload.get("top_k", 5)
    if isinstance(requested_top_k, bool) or not isinstance(requested_top_k, int):
        return jsonify({"error": "top_k must be an integer"}), 400
    top_k = max(1, min(requested_top_k, 5))

    result = current_model.predict(text, top_k=top_k)

    if not result:
        return jsonify({"predictions": [], "message": "No predictions available"})

    return jsonify({
        "input": text,
        "predictions": result,
        "top_word": result[0]["word"],
    })


@app.route("/api/retrain", methods=["POST"])
def retrain_model():
    global current_model
    sentences = load_corpus()
    new_model = BayesianNextWordModel(max_context=3, alpha=0.8)
    new_model.train(sentences)
    new_model.save_model(MODEL_PATH)
    current_model = new_model
    return jsonify({
        "status": "success",
        "message": "Model retrained successfully",
        "metadata": current_model.metadata(),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
