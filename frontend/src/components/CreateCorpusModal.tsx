import { useState, useRef, type DragEvent, type ChangeEvent } from "react";
import { slugify, generateId } from "../lib/slugify";
import { db } from "../lib/storage";
import { api } from "../lib/api";
import type { Corpus, FreshDocument } from "../lib/types";

interface CreateCorpusModalProps {
  onClose: () => void;
  onCreated: (corpusId: string) => void;
}

async function readFileText(file: File): Promise<string> {
  if ("text" in file && typeof file.text === "function") {
    return file.text();
  }
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(reader.error ?? new Error(`Failed to read ${file.name}`));
    reader.readAsText(file);
  });
}

const ACCEPTED_EXTENSIONS = [".txt", ".md", ".pdf", ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".c", ".cpp", ".h"];

export function CreateCorpusModal({ onClose, onCreated }: CreateCorpusModalProps) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [tagsInput, setTagsInput] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const droppedFiles = Array.from(e.dataTransfer.files).filter((f) =>
      ACCEPTED_EXTENSIONS.some((ext) => f.name.toLowerCase().endsWith(ext))
    );
    setFiles((prev) => [...prev, ...droppedFiles]);
  };

  const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files ?? []);
    setFiles((prev) => [...prev, ...selectedFiles]);
  };

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleCreate = async () => {
    if (!name.trim()) {
      setError("Name is required");
      return;
    }
    if (files.length === 0) {
      setError("At least one file is required");
      return;
    }

    const corpusId = slugify(name) || generateId();
    const existing = await db.corpora.get(corpusId);
    if (existing) {
      setError("A corpus with this name already exists");
      return;
    }

    setCreating(true);
    setError(null);

    try {
      const documents: FreshDocument[] = await Promise.all(
        files.map(async (file) => ({
          title: file.name,
          text: await readFileText(file),
          type: "code" as const,
          category: file.name.split(".").pop() || "text",
          year: new Date().getFullYear(),
          path: file.name,
          lang: file.name.split(".").pop() || null,
          repo: "uploaded",
        }))
      );

      const result = await api.ingest(corpusId, documents);

      const corpus: Corpus = {
        id: corpusId,
        name: name.trim(),
        description: description.trim(),
        tags: tagsInput.split(",").map((t) => t.trim()).filter(Boolean),
        createdAt: Date.now(),
        lastUsedAt: Date.now(),
        isFavorite: false,
        isDemo: false,
        documentCount: result.n_chunks,
        source: "files",
      };

      await db.corpora.put(corpus);
      onCreated(corpusId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create corpus");
      setCreating(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2 className="modal-title">Create New Corpus</h2>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>

        <div className="modal-body">
          <div className="form-field">
            <label className="form-label">Name *</label>
            <input
              type="text"
              className="form-input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My Research Papers"
              autoFocus
            />
          </div>

          <div className="form-field">
            <label className="form-label">Description</label>
            <textarea
              className="form-textarea"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What's in this corpus?"
              rows={3}
            />
          </div>

          <div className="form-field">
            <label className="form-label">Tags (comma-separated)</label>
            <input
              type="text"
              className="form-input"
              value={tagsInput}
              onChange={(e) => setTagsInput(e.target.value)}
              placeholder="research, ml, papers"
            />
          </div>

          <div className="form-field">
            <label className="form-label">Files *</label>
            <div
              className={`file-dropzone${dragging ? " dragging" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept={ACCEPTED_EXTENSIONS.join(",")}
                onChange={handleFileSelect}
                style={{ display: "none" }}
              />
              <span className="dropzone-text">
                {dragging ? "Drop files here" : "Drag files here or click to browse"}
              </span>
              <span className="dropzone-hint">
                Accepts: {ACCEPTED_EXTENSIONS.join(", ")}
              </span>
            </div>

            {files.length > 0 && (
              <ul className="file-list">
                {files.map((file, index) => (
                  <li key={index} className="file-item">
                    <span className="file-name">{file.name}</span>
                    <button className="file-remove" onClick={() => removeFile(index)}>&times;</button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {error && <p className="form-error">{error}</p>}
        </div>

        <div className="modal-footer">
          <button className="btn-secondary" onClick={onClose} disabled={creating}>
            Cancel
          </button>
          <button className="btn-primary" onClick={handleCreate} disabled={creating}>
            {creating ? "Creating..." : "Create Corpus"}
          </button>
        </div>
      </div>
    </div>
  );
}
