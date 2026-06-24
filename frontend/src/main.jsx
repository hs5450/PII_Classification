import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { AlertTriangle, CheckCircle2, Loader2, ScanLine, ShieldCheck, Sparkles } from "lucide-react";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function formatPercent(value) {
  if (value === null || value === undefined) {
    return "Unavailable";
  }
  return `${Math.round(value * 100)}%`;
}

function signalLabel(signal) {
  return signal.replace(/^has_/, "").replaceAll("_", " ");
}

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const canSubmit = text.trim().length > 0 && !isLoading;

  const status = useMemo(() => {
    if (!result) {
      return null;
    }
    return result.pii_risk
      ? {
          label: "PII risk detected",
          tone: "risk",
          icon: AlertTriangle,
        }
      : {
          label: "No PII risk detected",
          tone: "safe",
          icon: CheckCircle2,
        };
  }, [result]);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!canSubmit) {
      return;
    }

    setIsLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: text.trim() }),
      });

      const body = await response.json();
      if (!response.ok) {
        throw new Error(body.detail || "Prediction failed.");
      }

      setResult(body);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsLoading(false);
    }
  }

  const StatusIcon = status?.icon;
  const confidencePercent =
    result?.confidence === null || result?.confidence === undefined
      ? 0
      : Math.round(result.confidence * 100);

  return (
    <main className="app-shell">
      <div className="ambient-grid" aria-hidden="true" />
      <section className="workspace" aria-labelledby="page-title">
        <div className="heading-row">
          <div>
            <h1 id="page-title">PII Checker</h1>
          </div>
          <div className="brand-mark" aria-hidden="true">
            <ScanLine size={30} strokeWidth={2.2} />
          </div>
        </div>

        <form className="checker" onSubmit={handleSubmit}>
          <div className="form-header">
            <label htmlFor="pii-text">Text line</label>
            <span>{text.length}/1000</span>
          </div>
          <div className="input-row">
            <input
              id="pii-text"
              type="text"
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Enter one line to check for PII"
              maxLength={1000}
              autoComplete="off"
            />
            <button type="submit" disabled={!canSubmit}>
              {isLoading ? <Loader2 className="spin" size={18} /> : <ShieldCheck size={18} />}
              <span>Check</span>
            </button>
          </div>
        </form>

        {error ? <div className="message error">{error}</div> : null}

        {result && status ? (
          <section className={`result ${status.tone}`} aria-live="polite">
            <div className="result-heading">
              <div className="status-icon">
                <StatusIcon size={26} />
              </div>
              <div>
                <h2>{status.label}</h2>
                <p>Model label: {result.predicted_label}</p>
              </div>
            </div>

            <div className="metrics">
              <div>
                <span>Confidence</span>
                <strong>{formatPercent(result.confidence)}</strong>
                <div className="confidence-track" aria-hidden="true">
                  <div style={{ width: `${confidencePercent}%` }} />
                </div>
              </div>
              <div>
                <span>Length</span>
                <strong>{result.text_length} chars</strong>
              </div>
            </div>

            <div className="signals">
              <span>Matched signals</span>
              {result.matched_signals.length > 0 ? (
                <div className="signal-list">
                  {result.matched_signals.map((signal) => (
                    <strong key={signal}>{signalLabel(signal)}</strong>
                  ))}
                </div>
              ) : (
                <p>No regex PII signals matched.</p>
              )}
            </div>
          </section>
        ) : (
          <section className="empty-state" aria-live="polite">
            <div className="pulse-orbit" aria-hidden="true">
              <ShieldCheck size={28} />
            </div>
            <div>
              <h2>Ready to check</h2>
              <p>The text is sent to your local FastAPI server and is not stored by this app.</p>
            </div>
          </section>
        )}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
