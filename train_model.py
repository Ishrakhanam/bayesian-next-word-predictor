from __future__ import annotations

import random
from pathlib import Path

from predictor import BayesianNextWordModel

BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / "data" / "training_corpus.txt"
MODEL_PATH = BASE_DIR / "model" / "bayesian_model.pkl"


def build_default_corpus() -> list[str]:
    return [
        "I am going to school every morning.",
        "I am going to college to study engineering.",
        "They are going to the market before sunset.",
        "The manager is going to the office early today.",
        "We are going home after the meeting finishes.",
        "She is going to the hospital for a checkup.",
        "The children are going to the park with their parents.",
        "Please go to the library and return the book.",
        "The team is planning to visit the museum this weekend.",
        "I would like to travel to the coast with my friends.",
        "The teacher gives homework to all the students.",
        "Students study in the library every evening.",
        "The project is due next week for the final review.",
        "We need to finish the presentation before lunch.",
        "The machine learns from data every day.",
        "Natural language processing uses context to understand meaning.",
        "Bayesian statistics update beliefs with evidence.",
        "Machine learning models improve with more data.",
        "The application predicts the next word in real time.",
        "Users type sentences and expect helpful suggestions.",
        "The algorithm ranks the most probable candidates.",
        "The model estimates posterior probabilities from prior knowledge.",
        "A strong recommendation system uses context and frequency.",
        "The teacher asks students to submit the assignment soon.",
        "The manager calls the team to discuss the report.",
        "We are meeting at the office at nine in the morning.",
        "The medical team prepares for the emergency response.",
        "She visits the market to buy fresh vegetables.",
        "The airport is crowded during the holiday season.",
        "My father works in an office near the city center.",
        "The family drives home after dinner and rest.",
        "I enjoy reading books and writing notes at home.",
        "The children play in the garden after school.",
        "The software team builds a better application each month.",
        "The company launches a new product in the summer.",
        "The campus library opens after the lecture ends.",
        "The model becomes more accurate with larger data.",
        "The sentence shows the likely next word based on context.",
        "The dataset contains many examples of common phrases.",
        "Digital assistants suggest the next action quickly.",
        "The design includes a dark backdrop with pastel accents.",
        "The dashboard presents probability bars for each option.",
        "The system responds instantly while the user types.",
        "The backend trains the Bayesian model locally.",
        "The frontend displays predictions in a professional layout.",
        "The analysis focuses on word frequency and context.",
        "The user input is cleaned before inference begins.",
        "The model handles unknown contexts with smoothing.",
        "The probability values help users understand uncertainty.",
        "The training script saves the model to disk.",
        "The next word prediction system supports real time behavior.",
        "We prepare for the exam by reading the textbook.",
        "The developer writes code for the prediction engine.",
        "The system prioritizes the most likely word choices.",
        "The data pipeline stores a local english corpus.",
        "The application requires no paid external services.",
        "The language model uses learned word relationships.",
        "A consistent context keeps the prediction reliable.",
        "The project demonstrates NLP, bayesian inference, and ui design.",
        "The local dataset supports offline prediction at runtime.",
        "The script retrains the model when new corpus data arrives.",
        "The evaluation checks top one and top five accuracy.",
        "Students compare words through frequency and context.",
        "The model learns from repeated phrases in the corpus.",
        "The user can click a suggestion to continue typing.",
        "The interface updates suggestions while the sentence changes.",
    ]


def load_corpus() -> list[str]:
    if CORPUS_PATH.exists():
        content = CORPUS_PATH.read_text(encoding="utf-8")
        return [line.strip() for line in content.splitlines() if line.strip()]
    return build_default_corpus()


def split_corpus(
    sentences: list[str],
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if len(sentences) < 2:
        return list(sentences), []

    shuffled = list(sentences)
    random.Random(seed).shuffle(shuffled)
    split_index = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_fraction)))
    return shuffled[:split_index], shuffled[split_index:]


def evaluate_predictions(
    model: BayesianNextWordModel,
    sentences: list[str],
) -> dict[str, float | int]:
    total = 0
    correct_top1 = 0
    correct_top5 = 0

    for sentence in sentences:
        tokens = model.normalize_text(sentence)
        for index in range(1, len(tokens)):
            context = " ".join(tokens[max(0, index - model.max_context):index])
            true_word = tokens[index]
            predictions = model.predict(context, top_k=5)
            predicted_words = [item["word"] for item in predictions]
            total += 1
            if predicted_words and true_word == predicted_words[0]:
                correct_top1 += 1
            if true_word in predicted_words[:5]:
                correct_top5 += 1

    accuracy_top1 = (correct_top1 / total) if total else 0.0
    accuracy_top5 = (correct_top5 / total) if total else 0.0
    return {"top1_accuracy": accuracy_top1, "top5_accuracy": accuracy_top5, "evaluated_examples": total}


def evaluate_corpus(sentences: list[str]) -> dict[str, float | int]:
    train_sentences, test_sentences = split_corpus(sentences)
    evaluation_model = BayesianNextWordModel(max_context=3, alpha=0.8)
    evaluation_model.train(train_sentences)
    metrics = evaluate_predictions(evaluation_model, test_sentences)
    metrics.update({
        "training_sentences": len(train_sentences),
        "test_sentences": len(test_sentences),
        "training_vocabulary_size": len(evaluation_model.vocab),
    })
    return metrics


def main() -> None:
    sentences = load_corpus()
    model = BayesianNextWordModel(max_context=3, alpha=0.8)
    model.train(sentences)
    model.save_model(MODEL_PATH)

    metrics = evaluate_corpus(sentences)

    print("Model trained and saved to:", MODEL_PATH)
    print(f"Vocabulary size: {model.metadata()['vocabulary_size']}")
    print(f"Top-1 accuracy: {metrics['top1_accuracy']:.2%}")
    print(f"Top-5 accuracy: {metrics['top5_accuracy']:.2%}")
    print(
        f"Evaluated examples: {metrics['evaluated_examples']} "
        f"({metrics['training_sentences']} train / {metrics['test_sentences']} test sentences)"
    )


if __name__ == "__main__":
    main()
