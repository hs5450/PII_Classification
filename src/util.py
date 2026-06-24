from src.features import (
    add_regex_features,
    build_feature_frame,
    detect_regex_signals,
    regex_feature_names,
)
from src.inference import PiiPredictor
from src.modeling import encode_labels, train_model


def find_common_regex_patterns(df, text_col="text"):
    return add_regex_features(df, text_col=text_col)


__all__ = [
    "PiiPredictor",
    "add_regex_features",
    "build_feature_frame",
    "detect_regex_signals",
    "encode_labels",
    "find_common_regex_patterns",
    "regex_feature_names",
    "train_model",
]

