import numpy as np

from src.inference import PiiPredictor


class FakeClassifier:
    def predict(self, features):
        return np.array([1])

    def predict_proba(self, features):
        return np.array([[0.1, 0.9]])


class FakeEncoder:
    classes_ = np.array(["safe", "pii"])

    def inverse_transform(self, values):
        return np.array([self.classes_[values[0]]])


class FakeEmbeddingModel:
    def encode(self, texts):
        return np.array([[0.25, 0.75] for _ in texts])


def test_predict_returns_stable_response_shape():
    artifacts = {
        "classifier": FakeClassifier(),
        "label_encoder": FakeEncoder(),
        "feature_columns": [
            "has_email",
            "has_phone",
            "has_credit_card",
            "has_api_key",
            "has_password",
            "has_postcode",
            "embed_0",
            "embed_1",
        ],
    }
    predictor = PiiPredictor(artifacts=artifacts, embedding_model=FakeEmbeddingModel())

    result = predictor.predict("my email is hamza@example.com")

    assert result["pii_risk"] is True
    assert result["predicted_label"] == "pii"
    assert result["confidence"] == 0.9
    assert result["regex_signals"]["has_email"] is True
    assert "has_email" in result["matched_signals"]
    assert result["probabilities"] == {"safe": 0.1, "pii": 0.9}

