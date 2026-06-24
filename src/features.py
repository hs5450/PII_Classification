import re
from collections import OrderedDict

import numpy as np
import pandas as pd


REGEX_PATTERNS = OrderedDict(
    {
        "has_email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "has_phone": r"(\+44\s?7\d{3}|\(?07\d{3}\)?)\s?\d{3}\s?\d{3}",
        "has_credit_card": r"\b(?:\d[ -]*?){13,16}\b",
        "has_api_key": r"\b(sk_live|sk_test|ghp|AKIA|xoxb|whsec)_[A-Za-z0-9_\-]{8,}\b",
        "has_password": r"(?i)(password|passwd|pwd)\s*[:=]\s*\S+",
        "has_postcode": r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b",
    }
)


def regex_feature_names() -> list[str]:
    return list(REGEX_PATTERNS.keys())


def detect_regex_signals(text: str) -> dict[str, bool]:
    value = "" if text is None else str(text)
    return {
        feature: bool(re.search(pattern, value))
        for feature, pattern in REGEX_PATTERNS.items()
    }


def add_regex_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    result = df.copy()
    for feature in regex_feature_names():
        result[feature] = result[text_col].apply(
            lambda value, name=feature: int(detect_regex_signals(value)[name])
        )
    return result


def regex_features_for_text(text: str) -> pd.DataFrame:
    signals = detect_regex_signals(text)
    return pd.DataFrame([{name: int(signals[name]) for name in regex_feature_names()}])


def embeddings_to_frame(embeddings: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        embeddings,
        columns=[f"embed_{index}" for index in range(embeddings.shape[1])],
    )


def build_feature_frame(
    texts: list[str],
    embeddings: np.ndarray,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    base = pd.DataFrame({"text": texts})
    regex_df = add_regex_features(base)[regex_feature_names()]
    embedding_df = embeddings_to_frame(embeddings)
    features = pd.concat([regex_df.reset_index(drop=True), embedding_df], axis=1)
    if feature_columns is not None:
        features = features.reindex(columns=feature_columns, fill_value=0)
    return features

