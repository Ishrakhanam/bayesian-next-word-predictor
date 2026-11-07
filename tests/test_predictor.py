import math
import tempfile
import unittest
from pathlib import Path

from app import MODEL_PATH, app
from predictor import BayesianNextWordModel
from train_model import evaluate_corpus, split_corpus


class BayesianModelTests(unittest.TestCase):
    def setUp(self):
        self.model = BayesianNextWordModel(max_context=1, alpha=1.0)
        self.model.train(["red blue", "red blue", "red green"])

    def test_bayesian_prior_and_context_likelihood_match_counts(self):
        self.assertAlmostEqual(self.model._prior("blue"), 1 / 3)
        self.assertAlmostEqual(self.model._likelihood("blue", ("red",)), 3 / 4)
        self.assertAlmostEqual(self.model._likelihood("green", ("red",)), 2 / 3)

    def test_posterior_is_normalized_and_uses_context_likelihood(self):
        predictions = self.model.predict("red", top_k=3)
        scores = {
            word: self.model._prior(word) * self.model._likelihood(word, ("red",))
            for word in self.model.vocab
        }
        denominator = sum(scores.values())

        self.assertEqual(predictions[0]["word"], "blue")
        self.assertAlmostEqual(predictions[0]["probability"], scores["blue"] / denominator)
        self.assertAlmostEqual(sum(item["probability"] for item in predictions), 1.0)

    def test_unseen_context_backs_off_and_unknown_context_uses_prior(self):
        model = BayesianNextWordModel(max_context=2, alpha=0.8)
        model.train([
            "old red blue",
            "new red green",
            "old red blue",
            "new red green",
        ])

        backed_off = model.predict("unseen red", top_k=5)
        prior_predictions = model.predict("unknown context words", top_k=5)

        self.assertIn(backed_off[0]["word"], {"blue", "green"})
        self.assertEqual(prior_predictions, model.predict("", top_k=5))

    def test_predictions_have_finite_probabilities_and_are_top_five(self):
        predictions = self.model.predict("red", top_k=5)
        self.assertLessEqual(len(predictions), 5)
        self.assertTrue(all(math.isfinite(item["probability"]) for item in predictions))
        self.assertTrue(all(0 <= item["probability"] <= 1 for item in predictions))

    def test_model_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.pkl"
            self.model.save_model(path)
            restored = BayesianNextWordModel.load_model(path)

        self.assertEqual(restored.predict("red", 3), self.model.predict("red", 3))

    def test_retraining_replaces_previous_counts(self):
        self.model.train(["green blue"])
        fresh_model = BayesianNextWordModel(max_context=1, alpha=1.0)
        fresh_model.train(["green blue"])

        self.assertEqual(self.model.word_counts, fresh_model.word_counts)
        self.assertEqual(self.model.predict("green", 2), fresh_model.predict("green", 2))


class ProjectIntegrationTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_dashboard_and_read_endpoints(self):
        for path in ("/", "/api/health", "/api/model-info", "/api/corpus", "/api/metrics"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)

        dashboard = self.client.get("/").get_data(as_text=True)
        self.assertIn('data-page="predictor"', dashboard)
        self.assertIn('data-page="corpus"', dashboard)
        self.assertIn('data-page="metrics"', dashboard)

    def test_multiple_known_and_unseen_inputs_return_valid_posterior_values(self):
        for text in ("I am going to", "The teacher gives homework", "never-seen-word combination"):
            with self.subTest(text=text):
                response = self.client.post("/api/predict", json={"text": text, "top_k": 5})
                self.assertEqual(response.status_code, 200)
                predictions = response.get_json()["predictions"]
                self.assertEqual(len(predictions), 5)
                self.assertTrue(all(0 < item["probability"] <= 1 for item in predictions))
                self.assertLessEqual(sum(item["probability"] for item in predictions), 1.0)

    def test_predict_rejects_malformed_payloads(self):
        for payload in (
            {"text": "valid", "top_k": "not-an-int"},
            {"text": 42},
            ["not", "an", "object"],
        ):
            with self.subTest(payload=payload):
                response = self.client.post("/api/predict", json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())

    def test_retrain_endpoint_saves_loadable_model(self):
        response = self.client.post("/api/retrain")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "success")
        restored = BayesianNextWordModel.load_model(MODEL_PATH)
        self.assertEqual(restored.model_version, BayesianNextWordModel.MODEL_VERSION)

    def test_held_out_evaluation_uses_a_separate_deterministic_split(self):
        sentences = ["one two", "three four", "five six", "seven eight", "nine ten"]
        first_train, first_test = split_corpus(sentences)
        second_train, second_test = split_corpus(sentences)
        metrics = evaluate_corpus(sentences)

        self.assertEqual((first_train, first_test), (second_train, second_test))
        self.assertFalse(set(first_train) & set(first_test))
        self.assertEqual(metrics["training_sentences"], len(first_train))
        self.assertEqual(metrics["test_sentences"], len(first_test))
        self.assertGreater(metrics["evaluated_examples"], 0)


if __name__ == "__main__":
    unittest.main()
