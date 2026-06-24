from dataclasses import dataclass
from pathlib import Path

from src.features import build_feature_frame, detect_regex_signals
from src.modeling import DEFAULT_ARTIFACT_PATH, load_artifacts, load_embedding_model


SAFE_LABELS = {
    "safe",
    "clean",
    "benign",
    "normal",
    "none",
    "no_pii",
    "not_pii",
    "non_pii",
    "non-pii",
}


def _normalize_label(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_")


def label_implies_pii(label: str) -> bool:
    normalized = _normalize_label(label)
    return normalized not in SAFE_LABELS


@dataclass
class PiiPredictor:
    artifacts: dict
    embedding_model: object

    @classmethod
    def from_path(cls, artifact_path: str | Path = DEFAULT_ARTIFACT_PATH) -> "PiiPredictor":
        artifacts = load_artifacts(artifact_path)
        embedding_model_source = artifacts.get(
            "embedding_model_path",
            artifacts["embedding_model_name"],
        )
        embedding_model = load_embedding_model(embedding_model_source)
        return cls(artifacts=artifacts, embedding_model=embedding_model)

    def predict(self, text: str) -> dict:
        clean_text = "" if text is None else str(text).strip()
        signals = detect_regex_signals(clean_text)
        embeddings = self.embedding_model.encode([clean_text])
        features = build_feature_frame(
            [clean_text],
            embeddings,
            feature_columns=self.artifacts["feature_columns"],
        )

        classifier = self.artifacts["classifier"]
        encoder = self.artifacts["label_encoder"]
        prediction = classifier.predict(features)[0]
        predicted_label = str(encoder.inverse_transform([prediction])[0])

        confidence = None
        probabilities = None
        if hasattr(classifier, "predict_proba"):
            raw_probabilities = classifier.predict_proba(features)[0]
            confidence = float(max(raw_probabilities))
            probabilities = {
                str(label): float(probability)
                for label, probability in zip(encoder.classes_, raw_probabilities)
            }

        regex_risk = any(signals.values())
        model_risk = label_implies_pii(predicted_label)

        return {
            "text_length": len(clean_text),
            "pii_risk": bool(regex_risk or model_risk),
            "predicted_label": predicted_label,
            "confidence": confidence,
            "regex_signals": signals,
            "matched_signals": [name for name, matched in signals.items() if matched],
            "probabilities": probabilities,
        }
