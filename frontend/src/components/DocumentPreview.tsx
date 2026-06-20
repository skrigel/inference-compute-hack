import { useState } from "react";
import type { CachedScore } from "../lib/scoreCache";

interface DocumentPreviewProps {
  result: CachedScore;
  onClose: () => void;
}

export function DocumentPreview({ result, onClose }: DocumentPreviewProps) {
  const [showFull, setShowFull] = useState(false);

  if (showFull) {
    return <DocumentFullPreview result={result} onClose={onClose} onBack={() => setShowFull(false)} />;
  }

  return (
    <div className="preview-overlay" onClick={onClose}>
      <div className="preview-card" onClick={(e) => e.stopPropagation()}>
        <button className="preview-close" onClick={onClose} aria-label="Close">
          &times;
        </button>

        <div className="preview-header">
          <span className={`type-badge type-${result.meta.type ?? "code"}`}>
            {result.meta.type ?? "code"}
          </span>
          <h2 className="preview-title">{result.meta.title}</h2>
        </div>

        <div className="preview-meta-grid">
          <div className="preview-meta-item">
            <span className="preview-meta-label">Category</span>
            <span className="preview-meta-value">{result.meta.category ?? "—"}</span>
          </div>
          <div className="preview-meta-item">
            <span className="preview-meta-label">Year</span>
            <span className="preview-meta-value">{result.meta.year ?? "—"}</span>
          </div>
          {result.meta.path && (
            <div className="preview-meta-item">
              <span className="preview-meta-label">Path</span>
              <span className="preview-meta-value">{result.meta.path}</span>
            </div>
          )}
          {result.meta.repo && (
            <div className="preview-meta-item">
              <span className="preview-meta-label">Repo</span>
              <span className="preview-meta-value">{result.meta.repo}</span>
            </div>
          )}
          {result.meta.lang && (
            <div className="preview-meta-item">
              <span className="preview-meta-label">Language</span>
              <span className="preview-meta-value">{result.meta.lang}</span>
            </div>
          )}
        </div>

        <div className="preview-score">
          <span className="preview-score-label">Relevance Score</span>
          <div className="preview-score-bar">
            <div className="preview-score-fill" style={{ width: `${Math.round(result.score * 100)}%` }} />
          </div>
          <span className="preview-score-value">{result.score.toFixed(3)}</span>
        </div>

        <div className="preview-actions">
          <button className="btn-primary" onClick={() => setShowFull(true)}>
            Preview Document
          </button>
        </div>
      </div>
    </div>
  );
}

interface DocumentFullPreviewProps {
  result: CachedScore;
  onClose: () => void;
  onBack: () => void;
}

function DocumentFullPreview({ result, onClose, onBack }: DocumentFullPreviewProps) {
  return (
    <div className="preview-overlay" onClick={onClose}>
      <div className="preview-full-panel" onClick={(e) => e.stopPropagation()}>
        <div className="preview-full-header">
          <button className="preview-back" onClick={onBack} aria-label="Back">
            &larr; Back
          </button>
          <h2 className="preview-title">{result.meta.title}</h2>
          <button className="preview-close" onClick={onClose} aria-label="Close">
            &times;
          </button>
        </div>

        <div className="preview-full-meta">
          <span className={`type-badge type-${result.meta.type ?? "code"}`}>
            {result.meta.type ?? "code"}
          </span>
          <span>{result.meta.category ?? "—"}</span>
          <span>{result.meta.year ?? "—"}</span>
          {result.meta.path && <span>{result.meta.path}</span>}
        </div>

        <div className="preview-full-content">
          <p className="preview-placeholder">
            Document content preview not available.
            <br />
            <br />
            The full document text is stored in the backend and would require an additional API call to fetch.
          </p>
        </div>
      </div>
    </div>
  );
}
