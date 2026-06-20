import type { Hint } from "../hooks/useHints";

interface HintCardProps {
  hint: Hint;
  onDismiss: () => void;
}

export function HintCard({ hint, onDismiss }: HintCardProps) {
  return (
    <aside className="hint-card">
      <div className="hint-icon">
        <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16">
          <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z" clipRule="evenodd" />
        </svg>
      </div>
      <div className="hint-content">
        <strong className="hint-title">{hint.title}</strong>
        <p className="hint-body">{hint.body}</p>
      </div>
      <button className="hint-dismiss" onClick={onDismiss} aria-label="Dismiss hint">
        &times;
      </button>
    </aside>
  );
}
