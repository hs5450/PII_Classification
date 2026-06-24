from fastapi.testclient import TestClient

from src import api


class FakePredictor:
    def predict(self, text):
        return {
            "text_length": len(text),
            "pii_risk": True,
            "predicted_label": "pii",
            "confidence": 0.88,
            "regex_signals": {"has_email": True},
            "matched_signals": ["has_email"],
            "probabilities": {"safe": 0.12, "pii": 0.88},
        }


def test_health_endpoint():
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint(monkeypatch):
    api.get_predictor.cache_clear()
    monkeypatch.setattr(api, "get_predictor", lambda: FakePredictor())
    client = TestClient(api.app)

    response = client.post("/predict", json={"text": "email hamza@example.com"})

    assert response.status_code == 200
    body = response.json()
    assert body["pii_risk"] is True
    assert body["predicted_label"] == "pii"
    assert body["matched_signals"] == ["has_email"]

