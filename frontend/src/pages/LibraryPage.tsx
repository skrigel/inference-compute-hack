import { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { CorpusCard } from "../components/CorpusCard";
import { CreateCorpusModal } from "../components/CreateCorpusModal";
import { useCorpora } from "../hooks/useCorpora";

type SortOption = "recent" | "name" | "size";
type FilterOption = "all" | "favorites" | "files" | "demo";

export function LibraryPage() {
  const navigate = useNavigate();
  const { corpora, loading, refresh, toggleFavorite, deleteCorpus } = useCorpora();
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState<SortOption>("recent");
  const [filterBy, setFilterBy] = useState<FilterOption>("all");
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  const filteredCorpora = useMemo(() => {
    let result = [...corpora];

    // Filter
    if (filterBy === "favorites") {
      result = result.filter((c) => c.isFavorite);
    } else if (filterBy === "files") {
      result = result.filter((c) => c.source === "files");
    } else if (filterBy === "demo") {
      result = result.filter((c) => c.isDemo);
    }

    // Search
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      result = result.filter(
        (c) =>
          c.name.toLowerCase().includes(query) ||
          c.description.toLowerCase().includes(query) ||
          c.tags.some((t) => t.toLowerCase().includes(query))
      );
    }

    // Sort
    if (sortBy === "recent") {
      result.sort((a, b) => b.lastUsedAt - a.lastUsedAt);
    } else if (sortBy === "name") {
      result.sort((a, b) => a.name.localeCompare(b.name));
    } else if (sortBy === "size") {
      result.sort((a, b) => b.documentCount - a.documentCount);
    }

    return result;
  }, [corpora, searchQuery, sortBy, filterBy]);

  const handleOpenCorpus = (corpusId: string) => {
    navigate(`/search/${corpusId}`);
  };

  const handleDelete = async (corpusId: string) => {
    await deleteCorpus(corpusId);
    setDeleteConfirm(null);
  };

  const handleCreated = (corpusId: string) => {
    setShowCreateModal(false);
    refresh();
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
        <h1 className="page-title">Corpus Library</h1>
        <button className="btn-primary" onClick={() => setShowCreateModal(true)}>
          + New Corpus
        </button>
      </div>

      <div className="library-toolbar">
        <input
          type="text"
          className="search-input"
          placeholder="Search corpora..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        <select
          className="toolbar-select"
          value={filterBy}
          onChange={(e) => setFilterBy(e.target.value as FilterOption)}
        >
          <option value="all">All</option>
          <option value="favorites">Favorites</option>
          <option value="files">My Files</option>
          <option value="demo">Demo</option>
        </select>
        <select
          className="toolbar-select"
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as SortOption)}
        >
          <option value="recent">Recent</option>
          <option value="name">Name</option>
          <option value="size">Size</option>
        </select>
      </div>

      {filteredCorpora.length === 0 ? (
        <div className="empty-state">
          <p>No corpora found</p>
          {corpora.length === 0 && (
            <button className="btn-primary" onClick={() => setShowCreateModal(true)}>
              Create your first corpus
            </button>
          )}
        </div>
      ) : (
        <div className="corpus-list">
          {filteredCorpora.map((corpus) => (
            <CorpusCard
              key={corpus.id}
              corpus={corpus}
              variant="full"
              onClick={() => handleOpenCorpus(corpus.id)}
              onStar={() => toggleFavorite(corpus.id)}
              onEdit={() => {/* TODO: Edit modal */}}
              onDelete={() => setDeleteConfirm(corpus.id)}
            />
          ))}
        </div>
      )}

      {showCreateModal && (
        <CreateCorpusModal
          onClose={() => setShowCreateModal(false)}
          onCreated={handleCreated}
        />
      )}

      {deleteConfirm && (
        <div className="modal-overlay" onClick={() => setDeleteConfirm(null)}>
          <div className="modal-content modal-small" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2 className="modal-title">Delete Corpus?</h2>
            </div>
            <div className="modal-body">
              <p>This will permanently delete the corpus and all its saved queries.</p>
            </div>
            <div className="modal-footer">
              <button className="btn-secondary" onClick={() => setDeleteConfirm(null)}>
                Cancel
              </button>
              <button className="btn-danger" onClick={() => handleDelete(deleteConfirm)}>
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
