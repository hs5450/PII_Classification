from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.features import build_feature_frame


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_ARTIFACT_PATH = Path("models/pii_classifier.joblib")
DEFAULT_EMBEDDING_MODEL_PATH = Path("models/embedding_model")


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def encode_labels(labels: pd.Series) -> tuple[pd.Series, LabelEncoder, dict[str, int]]:
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    return y, encoder, label_mapping


def train_model(
    csv_path: str | Path = "synthetic_sensitive_prompt_dataset.csv",
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    embedding_model_path: str | Path = DEFAULT_EMBEDDING_MODEL_PATH,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    test_size: float = 0.2,
    random_state: int = 1,
) -> dict:
    csv_path = Path(csv_path)
    artifact_path = Path(artifact_path)
    embedding_model_path = Path(embedding_model_path)

    data = pd.read_csv(csv_path)
    required_columns = {"id", "text", "label"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Training CSV is missing required columns: {missing}")

    embedding_model = load_embedding_model(embedding_model_name)
    embedding_model_path.mkdir(parents=True, exist_ok=True)
    embedding_model.save(str(embedding_model_path))

    texts = data["text"].fillna("").astype(str).tolist()
    embeddings = embedding_model.encode(texts)
    X = build_feature_frame(texts, embeddings)
    y, encoder, label_mapping = encode_labels(data["label"])

    stratify = y if len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred).tolist()

    artifacts = {
        "classifier": classifier,
        "label_encoder": encoder,
        "feature_columns": X.columns.tolist(),
        "embedding_model_name": embedding_model_name,
        "embedding_model_path": str(embedding_model_path),
        "label_mapping": label_mapping,
        "metrics": {
            "classification_report": report,
            "confusion_matrix": matrix,
        },
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, artifact_path)
    return artifacts


def load_artifacts(artifact_path: str | Path = DEFAULT_ARTIFACT_PATH) -> dict:
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {artifact_path}. "
            "Run `python -m src.train` first."
        )
    return joblib.load(artifact_path)
