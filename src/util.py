import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer

## Models:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report, confusion_matrix
#

def find_common_regex_patterns(df, text_col = "text"):
    # Phone, Email, address, credit card, api key, password ,postcode
    patterns = {
        "has_email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "has_phone": r"(\+44\s?7\d{3}|\(?07\d{3}\)?)\s?\d{3}\s?\d{3}",
        "has_credit_card": r"\b(?:\d[ -]*?){13,16}\b",
        "has_api_key": r"\b(sk_live|sk_test|ghp|AKIA|xoxb|whsec)_[A-Za-z0-9_\-]{8,}\b",
        "has_password": r"(?i)(password|passwd|pwd)\s*[:=]\s*\S+",
        "has_postcode": r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b"
    }

    for feature, pattern in patterns.items():
        df[feature] = df[text_col].apply(lambda x:1 if re.search(pattern, str(x)) else 0)
    return df

def embedding_output(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["text"].tolist())
    return embeddings
def classifier(X, y):
    clf = LogisticRegression(max_iter = 1000)
    clf.fit(X, y)
    return clf


def encode_labels(df, label_col="label"):
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[label_col])

    label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    return y, encoder, label_mapping

def make_stratified_split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def evaluate_model(y_true, y_pred, encoder=None):
    if encoder is not None:
        class_names = encoder.classes_
    else:
        class_names = None

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)

    if class_names is not None:
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    else:
        cm_df = pd.DataFrame(cm)

    print("\nConfusion Matrix:")
    print(cm_df)

    return cm_df

data = pd.read_csv("synthetic_sensitive_prompt_dataset.csv")
data = find_common_regex_patterns(data)
embeddings = embedding_output(data)
embeddings_df = pd.DataFrame(embeddings, columns = [f"embed_{i}" for i in range(embeddings.shape[1])])

final_df = pd.concat([data, embeddings_df], axis = 1)
drop_cols = ["id", "text"]
final_df = final_df.drop(columns = drop_cols)
X = final_df.drop(columns = ["label"])
y, encoder, label_mapping = encode_labels(final_df, label_col="label")
# print(label_mapping)
X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=1,
        stratify=y
    )
# print(X_train)
model = classifier(X_train, y_train)

y_pred = model.predict(X_test)

cm_df = evaluate_model(y_test, y_pred, encoder)