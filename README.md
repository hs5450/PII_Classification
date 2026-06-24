# PII Classification Website

Local demo app for checking whether a single text line contains or suggests PII.

The model follows the original notebook approach:

- regex features for common PII patterns
- `SentenceTransformer("all-MiniLM-L6-v2")` embeddings
- `LogisticRegression` classifier

## Project Layout

```text
src/
  api.py          FastAPI app
  features.py    Regex and feature-frame helpers
  inference.py   Saved-model prediction logic
  modeling.py    Training and artifact loading
  train.py       Training CLI
frontend/        Vite React UI
tests/           Backend tests
main.ipynb       Original experiment notebook
```

## Setup

Create and activate a Python environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

Place `synthetic_sensitive_prompt_dataset.csv` in the project root. It is ignored by git and should have these columns:

```text
id,text,label
```

## Train The Model

```powershell
python -m src.train --data synthetic_sensitive_prompt_dataset.csv
```

This writes the model artifact to:

```text
models/pii_classifier.joblib
```

It also saves the local embedding model files to:

```text
models/embedding_model
```

If you trained before this folder existed, run the training command again so the API can load embeddings locally without contacting Hugging Face during prediction.

## Run The Backend

```powershell
uvicorn src.api:app --reload
```

Health check:

```text
http://localhost:8000/health
```

Prediction endpoint:

```http
POST http://localhost:8000/predict
Content-Type: application/json

{ "text": "my email is jack@example.com" }
```

## Run The React Frontend

Install Node.js first if `node --version` and `npm --version` do not work.

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL, usually:

```text
http://localhost:5173
```

## Privacy

The app does not store submitted text. The frontend keeps only the current input/result in memory, and the backend does not persist request bodies.

## Tests

```powershell
pytest
```
