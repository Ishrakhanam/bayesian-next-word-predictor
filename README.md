# Bayesian Next Word Predictor

A complete Python NLP project that predicts the next word using a Bayesian categorical model trained on a local English-language corpus. The project includes a Flask backend, a dark pastel dashboard UI, and a reusable training pipeline.

## Features

- Real-time next-word prediction using Bayesian posterior probabilities
- Local English training corpus with offline model storage
- Flask REST API for prediction and retraining
- Professional dark-theme UI with pastel accents
- Model evaluation and training script
- Click-to-insert suggestions in the frontend

## Project structure

- `app.py` – Flask application entry point
- `predictor.py` – Bayesian model implementation
- `train_model.py` – Training and evaluation script
- `data/training_corpus.txt` – Local corpus used for model training
- `model/bayesian_model.pkl` – Saved trained model
- `templates/index.html` – Frontend dashboard
- `static/css/styles.css` – Dark pastel styling
- `static/js/app.js` – Frontend prediction logic

## Bayesian formulation

For a candidate word `W` and a selected suffix context `C`, the model applies Bayes' theorem:

P(W | C) = [P(C | W) × P(W)] / P(C)

The model estimates:

- W is the candidate next word
- `P(W) = (count(W) + alpha) / (total_tokens + alpha * vocabulary_size)`, the smoothed corpus word prior
- For a context of length `L`, `P(C | W) = (count(C, W) + alpha) / (context_count_L(W) + alpha * (observed_contexts_L + 1))`
- The extra context category in the likelihood denominator represents any unseen context of that length
- `P(C)` is the normalization constant; the posterior scores are normalized across the full vocabulary

The likelihood is estimated from the frequency of context `C` immediately preceding candidate `W`, conditioned on `W`; it is not the conditional next-word frequency `P(W | C)`. The model uses the longest observed suffix of the typed context, backs off to shorter suffixes when necessary, and falls back to the smoothed word prior when no suffix has been observed. The interface displays up to five candidates with their posterior probabilities; because probabilities are normalized over the entire vocabulary, the displayed top five need not sum to 100%.

## Running the app

1. Create and activate a virtual environment (Windows PowerShell):
   `py -3.13 -m venv .venv`
   `.\.venv\Scripts\Activate.ps1`
2. Install dependencies:
   `python -m pip install -r requirements.txt`
3. Train the model:
   `python train_model.py`
4. Start the server:
   `python app.py`
5. Open `http://127.0.0.1:5000` in the browser.

Run the automated verification suite with:
`python -m unittest discover -s tests -v`

## API endpoints

- `GET /` – dashboard UI
- `GET /api/health` – server health information
- `GET /api/model-info` – vocabulary metadata
- `GET /api/corpus` – local training sentences
- `GET /api/metrics` – deterministic held-out top-1 and top-5 evaluation
- `POST /api/predict` – returns top predictions for the given text
- `POST /api/retrain` – retrains the model and saves it to disk

## Model evaluation

The training script and Metrics dashboard use the same deterministic, seeded 80/20 sentence split. Only the training partition is used to fit the evaluation model; reported metrics are computed against the held-out test partition. The final deployed model is then retrained on the complete corpus. Evaluation reports:

- Top-1 accuracy
- Top-5 accuracy
- Number of evaluated samples

The bundled corpus is small and intended for demonstrating the workflow; its measured accuracy is not a general benchmark of English next-word prediction.

## Notes

- The project is fully local and does not require paid cloud AI services.
- The model is trained on a built-in English corpus so it works without internet access after installation.
- Predictions are generated dynamically from the trained model rather than hardcoded responses.
