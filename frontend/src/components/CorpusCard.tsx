import type { Corpus } from "../lib/types";

interface CorpusCardProps {
  corpus: Corpus;
  variant?: "compact" | "full";
  onClick?: () => void;
  onStar?: () => void;
  onEdit?: () => void;
  onDelete?: () => void;
}

function formatRelativeTime(timestamp: number): string {
  const seconds = Math.floor((Date.now() - timestamp) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(timestamp).toLocaleDateString();
}

export function CorpusCard({ corpus, variant = "full", onClick, onStar, onEdit, onDelete }: CorpusCardProps) {
  const isCompact = variant === "compact";

  return (
    <article className={`corpus-card ${variant}`} onClick={onClick}>
      <div className="corpus-card-header">
        <button
          className={`star-btn${corpus.isFavorite ? " active" : ""}`}
          onClick={(e) => { e.stopPropagation(); onStar?.(); }}
          aria-label={corpus.isFavorite ? "Remove from favorites" : "Add to favorites"}
          disabled={corpus.isDemo}
        >
          {corpus.isFavorite ? "\u2605" : "\u2606"}
        </button>
        <h3 className="corpus-card-title">
          {corpus.name}
          {corpus.isDemo && <span className="demo-badge">built-in</span>}
        </h3>
        <span className="corpus-card-count">{corpus.documentCount} docs</span>
      </div>

      {!isCompact && corpus.description && (
        <p className="corpus-card-desc">{corpus.description}</p>
      )}

      <div className="corpus-card-footer">
        {corpus.tags.length > 0 && (
          <div className="corpus-card-tags">
            {corpus.tags.slice(0, 3).map((tag) => (
              <span key={tag} className="corpus-tag">#{tag}</span>
            ))}
          </div>
        )}
        <span className="corpus-card-time">Last used: {formatRelativeTime(corpus.lastUsedAt)}</span>
      </div>

      {!isCompact && !corpus.isDemo && (
        <div className="corpus-card-actions" onClick={(e) => e.stopPropagation()}>
          <button className="btn-secondary" onClick={onEdit}>Edit</button>
          <button className="btn-danger" onClick={onDelete}>Delete</button>
        </div>
      )}
    </article>
  );
}
