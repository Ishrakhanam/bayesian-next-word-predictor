from __future__ import annotations

import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


class BayesianNextWordModel:
    MODEL_VERSION = 2

    def __init__(self, max_context: int = 3, alpha: float = 0.8):
        if max_context < 1:
            raise ValueError("max_context must be at least 1")
        if alpha <= 0:
            raise ValueError("alpha must be greater than 0")

        self.max_context = max_context
        self.alpha = alpha
        self.model_version = self.MODEL_VERSION
        self.word_counts: Counter[str] = Counter()
        self.context_counts: defaultdict[Tuple[str, ...], int] = defaultdict(int)
        self.pair_counts: defaultdict[Tuple[Tuple[str, ...], str], int] = defaultdict(int)
        self.word_context_counts: Counter[Tuple[int, str]] = Counter()
        self.context_vocab_by_length: defaultdict[int, Set[Tuple[str, ...]]] = defaultdict(set)
        self.vocab: List[str] = []
        self.total_tokens = 0
        self.total_sentences = 0
        self.training_summary: Dict[str, int | float] = {}

    @staticmethod
    def normalize_text(text: str) -> List[str]:
        cleaned = text.lower()
        tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", cleaned)
        return tokens

    def train(self, sentences: Iterable[str]) -> None:
        self.word_counts.clear()
        self.context_counts.clear()
        self.pair_counts.clear()
        self.word_context_counts.clear()
        self.context_vocab_by_length.clear()

        normalized_sentences: List[List[str]] = []
        for sentence in sentences:
            tokens = self.normalize_text(sentence)
            if len(tokens) >= 2:
                normalized_sentences.append(tokens)

        self.total_sentences = len(normalized_sentences)
        for tokens in normalized_sentences:
            self.word_counts.update(tokens)
            for index, target in enumerate(tokens):
                for size in range(1, min(self.max_context, index) + 1):
                    context = tuple(tokens[index - size:index])
                    self.context_vocab_by_length[size].add(context)
                    self.word_context_counts[(size, target)] += 1
                    self.context_counts[context] += 1
                    self.pair_counts[(context, target)] += 1

        self.vocab = sorted(self.word_counts)
        self.total_tokens = sum(self.word_counts.values())
        self.training_summary = {
            "vocabulary_size": len(self.vocab),
            "total_tokens": self.total_tokens,
            "sentence_count": self.total_sentences,
            "context_window": self.max_context,
            "smoothing_alpha": self.alpha,
        }

    def _prior(self, word: str) -> float:
        total = self.total_tokens + self.alpha * len(self.vocab)
        if total <= 0:
            return 0.0
        return (self.word_counts.get(word, 0) + self.alpha) / total

    def _likelihood(self, word: str, context: Tuple[str, ...]) -> float:
        context_size = len(context)
        matching_context_count = self.pair_counts.get((context, word), 0)
        word_context_total = self.word_context_counts.get((context_size, word), 0)
        context_categories = len(self.context_vocab_by_length[context_size]) + 1
        return (matching_context_count + self.alpha) / (
            word_context_total + self.alpha * context_categories
        )

    def predict(self, text: str, top_k: int = 5) -> List[Dict[str, float | str]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self.vocab:
            return []

        tokens = self.normalize_text(text)
        context = None
        for size in range(min(len(tokens), self.max_context), 0, -1):
            candidate_context = tuple(tokens[-size:])
            if self.context_counts.get(candidate_context, 0):
                context = candidate_context
                break

        if context is None:
            candidate_scores = {word: self._prior(word) for word in self.vocab}
        else:
            candidate_scores = {
                word: self._prior(word) * self._likelihood(word, context)
                for word in self.vocab
            }

        total_score = sum(candidate_scores.values())
        if total_score <= 0:
            return []

        ranked = [
            {"word": word, "probability": candidate_scores[word] / total_score}
            for word in sorted(
                self.vocab,
                key=lambda word: (-candidate_scores[word], word),
            )
        ]

        return ranked[:top_k]

    def save_model(self, path: str | Path) -> None:
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load_model(cls, path: str | Path) -> "BayesianNextWordModel":
        model_path = Path(path)
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
        return model

    def metadata(self) -> Dict[str, int | float]:
        return {
            "vocabulary_size": len(self.vocab),
            "total_tokens": self.total_tokens,
            "context_window": self.max_context,
            "alpha": self.alpha,
            "sentence_count": self.total_sentences,
        }
