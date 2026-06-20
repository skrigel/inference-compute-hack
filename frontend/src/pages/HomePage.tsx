import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { CorpusCard } from "../components/CorpusCard";
import { useCorpora } from "../hooks/useCorpora";
import { db } from "../lib/storage";

const BUILT_IN_CORPORA = [
  {
    id: "demo",
    name: "Demo Corpus",
    description: "7 code snippets for quick testing",
    docCount: 7,
  },
  {
    id: "browsecomp",
    name: "BrowseComp+",
    description: "1,000 web documents for semantic search",
    docCount: 1000,
  },
];

export function HomePage() {
  const navigate = useNavigate();
  const { corpora, favorites, recent, loading, toggleFavorite } = useCorpora();
  const [queryCount, setQueryCount] = useState(0);

  useEffect(() => {
    db.savedQueries.countAll().then(setQueryCount);
  }, []);

  const handleOpenCorpus = (corpusId: string) => {
    navigate(`/search/${corpusId}`);
  };

  if (loading) {
    return (
      <main className="page-content">
        <div className="loading-state">Loading...</div>
      </main>
    );
  }

  return (
    <main className="page-content">
      <div className="page-header">
        <h1 className="page-title">Welcome back</h1>
        <button className="btn-primary" onClick={() => navigate("/library")}>
          + New Corpus
        </button>
      </div>

      <section className="builtin-corpora-section">
        <h2 className="section-title">Built-in Corpora</h2>
        <div className="builtin-corpora-grid">
          {BUILT_IN_CORPORA.map((corpus) => (
            <article
              key={corpus.id}
              className="builtin-corpus-card"
              onClick={() => handleOpenCorpus(corpus.id)}
            >
              <div className="builtin-corpus-header">
                <span className="builtin-corpus-name">{corpus.name}</span>
                <span className="builtin-corpus-badge">{corpus.docCount} docs</span>
              </div>
              <p className="builtin-corpus-description">{corpus.description}</p>
              <button className="btn-secondary">Open</button>
            </article>
          ))}
        </div>
      </section>

      <div className="dashboard-grid">
        <section className="dashboard-section favorites-section">
          <h2 className="section-title">Favorites</h2>
          {favorites.length === 0 ? (
            <p className="empty-hint">Star your frequently used corpora to pin them here</p>
          ) : (
            <div className="favorites-row">
              {favorites.map((corpus) => (
                <CorpusCard
                  key={corpus.id}
                  corpus={corpus}
                  variant="compact"
                  onClick={() => handleOpenCorpus(corpus.id)}
                  onStar={() => toggleFavorite(corpus.id)}
                />
              ))}
            </div>
          )}
        </section>

        <section className="dashboard-section recent-section">
          <h2 className="section-title">Recent</h2>
          {recent.length === 0 ? (
            <p className="empty-hint">No recent corpora</p>
          ) : (
            <ul className="recent-list">
              {recent.map((corpus) => (
                <li
                  key={corpus.id}
                  className="recent-item"
                  onClick={() => handleOpenCorpus(corpus.id)}
                >
                  <span className="recent-name">{corpus.name}</span>
                  <span className="recent-time">
                    {formatRelativeTime(corpus.lastUsedAt)}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>

      <section className="stats-row">
        <span className="stat">{corpora.length} corpora</span>
        <span className="stat-sep">&middot;</span>
        <span className="stat">{favorites.length} favorites</span>
        <span className="stat-sep">&middot;</span>
        <span className="stat">{queryCount} saved queries</span>
      </section>

      <section className="quick-start">
        <h2 className="section-title">Quick Actions</h2>
        <div className="quick-actions">
          <button className="btn-quick" onClick={() => navigate("/library")}>
            Upload Files
          </button>
          <button className="btn-quick-link" onClick={() => navigate("/library")}>
            View Library &rarr;
          </button>
        </div>
      </section>
    </main>
  );
}

function formatRelativeTime(timestamp: number): string {
  const seconds = Math.floor((Date.now() - timestamp) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return days === 1 ? "yesterday" : `${days}d ago`;
  return new Date(timestamp).toLocaleDateString();
}
