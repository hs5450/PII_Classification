import argparse
from pathlib import Path

from src.modeling import DEFAULT_ARTIFACT_PATH, DEFAULT_EMBEDDING_MODEL, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PII classifier.")
    parser.add_argument(
        "--data",
        default="synthetic_sensitive_prompt_dataset.csv",
        help="Path to the training CSV with id, text, and label columns.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_ARTIFACT_PATH),
        help="Where to save the trained model artifact.",
    )
    parser.add_argument(
        "--embedding-output",
        default="models/embedding_model",
        help="Where to save the local SentenceTransformer files used for inference.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = train_model(
        csv_path=args.data,
        artifact_path=args.output,
        embedding_model_path=args.embedding_output,
        embedding_model_name=args.embedding_model,
    )
    print(f"Saved model artifact to {Path(args.output)}")
    print(f"Saved embedding model to {Path(args.embedding_output)}")
    print(f"Labels: {artifacts['label_mapping']}")


if __name__ == "__main__":
    main()
