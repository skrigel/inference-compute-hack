import {
  ChangeEvent,
  DragEvent,
  FormEvent,
  PointerEvent as ReactPointerEvent,
  useEffect,
  useRef,
  useState,
} from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useDashboard, type LatencyKind } from "../hooks/useDashboard";
import { db } from "../lib/storage";
import type { CachedScore } from "../lib/scoreCache";
import type { Chip, FacetBucket, Facets, HistogramBin, Corpus } from "../lib/types";

const DEFAULT_QUERY = "every place we retry a network call without backoff";

export function SearchPage() {
  const { corpusId } = useParams<{ corpusId: string }>();
  const navigate = useNavigate();
  const [corpus, setCorpus] = useState<Corpus | null>(null);
  const [loading, setLoading] = useState(true);
  const d = useDashboard(DEFAULT_QUERY);

  useEffect(() => {
    if (!corpusId) {
      navigate("/");
      return;
    }

    db.corpora.get(corpusId).then((c) => {
      if (!c) {
        navigate("/library");
        return;
      }
      setCorpus(c);
      setLoading(false);

      // Update lastUsedAt
      db.corpora.put({ ...c, lastUsedAt: Date.now() });
      db.preferences.set("lastCorpusId", corpusId);
    });
  }, [corpusId, navigate]);

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void d.runQuery(d.predicate);
  };

  if (loading) {
    return (
      <main className="page-content">
        <div className="loading-state">Loading corpus...</div>
      </main>
    );
  }

  return (
    <main className="search-page">
      <QuerySection
        predicate={d.predicate}
        onPredicateChange={d.setPredicate}
        onSubmit={onSubmit}
        streaming={d.streaming}
        corpus={corpus}
      />

      <ThresholdSection
        histogram={d.view.histogram}
        threshold={d.threshold}
        onThreshold={d.setThreshold}
        hasRun={d.hasRun}
        matched={d.view.matched}
        total={d.scanned}
      />

      <FilterSection
        chips={d.chips}
        refining={d.refining}
        onRefine={d.runRefine}
        onRemoveChip={d.removeChip}
        onFreshFiles={d.ingestFreshFiles}
      />

      <TabbedSection
        activeTab="results"
        hasRun={d.hasRun}
        results={d.view.results}
        facets={d.view.facets}
        threshold={d.threshold}
        matched={d.view.matched}
        etaMs={d.etaMs}
        latencyMs={d.latencyMs}
        latencyKind={d.latencyKind}
        onClickRefine={d.runClickRefine}
      />
    </main>
  );
}

interface QuerySectionProps {
  predicate: string;
  onPredicateChange: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  streaming: boolean;
  corpus: Corpus | null;
}

function QuerySection({ predicate, onPredicateChange, onSubmit, streaming, corpus }: QuerySectionProps) {
  return (
    <section className="query-section">
      <div className="query-header">
        <span className="corpus-name">{corpus?.name ?? "Unknown Corpus"}</span>
        <span className="corpus-docs">{corpus?.documentCount ?? 0} documents</span>
      </div>
      <form className="query-form" onSubmit={onSubmit}>
        <input
          aria-label="Search query"
          autoComplete="off"
          value={predicate}
          onChange={(event) => onPredicateChange(event.target.value)}
          placeholder="Describe what you're looking for..."
        />
        <button className="btn-primary" type="submit" disabled={streaming}>
          {streaming ? "Scanning..." : "Scan"}
        </button>
      </form>
    </section>
  );
}

interface ThresholdSectionProps {
  histogram: HistogramBin[];
  threshold: number;
  onThreshold: (value: number) => void;
  hasRun: boolean;
  matched: number;
  total: number;
}

function ThresholdSection({ histogram, threshold, onThreshold, hasRun, matched, total }: ThresholdSectionProps) {
  return (
    <section className="threshold-section">
      <div className="threshold-header">
        <span className="threshold-label">Relevance Threshold</span>
        <span className="threshold-stats">
          <strong>{matched}</strong> of {total} <span className="threshold-value">{threshold.toFixed(2)}</span>
        </span>
      </div>
      <Histogram histogram={histogram} threshold={threshold} onThreshold={onThreshold} hasRun={hasRun} />
    </section>
  );
}

interface HistogramProps {
  histogram: HistogramBin[];
  threshold: number;
  onThreshold: (value: number) => void;
  hasRun: boolean;
}

function Histogram({ histogram, threshold, onThreshold, hasRun }: HistogramProps) {
  const ref = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const max = Math.max(1, ...histogram.map((bin) => bin.count));

  const setFromClientX = (clientX: number) => {
    const element = ref.current;
    if (!element) return;
    const rect = element.getBoundingClientRect();
    onThreshold(Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)));
  };

  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!hasRun) return;
    dragging.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    setFromClientX(event.clientX);
  };

  return (
    <div
      className={`histogram${!hasRun ? " empty" : ""}`}
      ref={ref}
      onPointerDown={onPointerDown}
      onPointerMove={(event) => dragging.current && setFromClientX(event.clientX)}
      onPointerUp={() => (dragging.current = false)}
      tabIndex={0}
    >
      <div className="histogram-bars">
        {histogram.map((bin, index) => {
          const center = (index + 0.5) / histogram.length;
          return (
            <div className={`bin${center >= threshold ? " in" : ""}`} key={index}>
              <div className="fill" style={{ height: hasRun ? `${(bin.count / max) * 100}%` : "0%" }} />
            </div>
          );
        })}
      </div>
      <div className="threshold-thumb" style={{ left: `calc(${threshold * 100}% + 12px)` }} />
      <div className="histogram-axis">
        <span>0</span>
        <span>1</span>
      </div>
    </div>
  );
}

interface FilterSectionProps {
  chips: Chip[];
  refining: boolean;
  onRefine: (utterance: string) => Promise<void>;
  onRemoveChip: (clauseId: string) => Promise<void>;
  onFreshFiles: (files: File[] | FileList) => Promise<void>;
}

function FilterSection({ chips, refining, onRefine, onRemoveChip, onFreshFiles }: FilterSectionProps) {
  const [utterance, setUtterance] = useState("");
  const [dragging, setDragging] = useState(false);

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const next = utterance.trim();
    if (!next) return;
    setUtterance("");
    void onRefine(next);
  };

  const ingest = (files: FileList | null) => {
    if (!files?.length) return;
    void onFreshFiles(files);
  };

  const onDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragging(false);
    ingest(event.dataTransfer.files);
  };

  return (
    <section className="filter-section">
      <div className="filter-bar">
        <div className="chip-rail">
          {chips.map((chip) => (
            <button className="chip" key={chip.clause_id} onClick={() => void onRemoveChip(chip.clause_id)}>
              <span className="chip-label">{chip.label}</span>
              <span className="chip-text">{chip.text}</span>
              <span className="chip-remove">&times;</span>
            </button>
          ))}

          <form className="add-filter-form" onSubmit={submit}>
            <input
              aria-label="Refine query"
              value={utterance}
              onChange={(event) => setUtterance(event.target.value)}
              placeholder="Refine: only python, without tests..."
            />
            <button className="btn-add" type="submit" disabled={refining || !utterance.trim()}>
              {refining ? "..." : "Add"}
            </button>
          </form>

          <label
            className={`dropzone${dragging ? " active" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <input
              type="file"
              multiple
              onChange={(event: ChangeEvent<HTMLInputElement>) => ingest(event.target.files)}
            />
            + files
          </label>
        </div>
      </div>
    </section>
  );
}

interface TabbedSectionProps {
  activeTab: string;
  hasRun: boolean;
  results: CachedScore[];
  facets: Facets;
  threshold: number;
  matched: number;
  etaMs: number;
  latencyMs: number;
  latencyKind: LatencyKind;
  onClickRefine: (chunkId: string, sign: "+" | "-") => Promise<void>;
}

function TabbedSection({ hasRun, results, facets, threshold, matched, etaMs, latencyMs, latencyKind, onClickRefine }: TabbedSectionProps) {
  const [tab, setTab] = useState<"results" | "facets">("results");

  return (
    <section className="tabbed-section">
      <div className="tab-bar">
        <button className={`tab-btn${tab === "results" ? " active" : ""}`} onClick={() => setTab("results")}>
          Results <span className="tab-count">{matched}</span>
        </button>
        <button className={`tab-btn${tab === "facets" ? " active" : ""}`} onClick={() => setTab("facets")}>
          Facets
        </button>
        <div className="tab-bar-metrics">
          {etaMs > 0 && <span className="eta-tag">ETA {etaMs}ms</span>}
          <span className={`latency-tag lat-${latencyKind}`}>
            <span className="lat-kind">{latencyKind}</span> {Math.round(latencyMs)}ms
          </span>
        </div>
      </div>

      <div className="tab-panel">
        {tab === "results" && (
          <ResultList results={results} threshold={threshold} hasRun={hasRun} onClickRefine={onClickRefine} />
        )}
        {tab === "facets" && hasRun && <FacetsPanel facets={facets} />}
      </div>
    </section>
  );
}

function ResultList({
  results,
  threshold,
  hasRun,
  onClickRefine,
}: {
  results: CachedScore[];
  threshold: number;
  hasRun: boolean;
  onClickRefine: (chunkId: string, sign: "+" | "-") => Promise<void>;
}) {
  if (!hasRun) {
    return (
      <div className="panel-empty">
        <h3>No scan yet</h3>
        <p>Type a query and click Scan to search this corpus.</p>
      </div>
    );
  }

  if (!results.length) {
    return (
      <div className="panel-empty">
        <h3>No results</h3>
        <p>Adjust the threshold or refine your query.</p>
      </div>
    );
  }

  return (
    <div className="results-list">
      {results.map((result) => {
        const matched = result.score >= threshold;
        return (
          <article className={`result-row${matched ? " matched" : ""}`} key={result.chunk_id}>
            <div className="result-score">
              <div className="score-value">{result.score.toFixed(2)}</div>
              <div className="score-bar">
                <div className="score-fill" style={{ width: `${Math.round(result.score * 100)}%` }} />
              </div>
            </div>
            <div className="result-content">
              <div className="result-header">
                <span className={`type-badge type-${result.meta.type ?? "code"}`}>
                  {result.meta.type ?? "code"}
                </span>
                <span className="result-title">{result.meta.title}</span>
              </div>
              <div className="result-meta">
                {result.meta.category ?? "—"} · {result.meta.year ?? "—"}
              </div>
              <div className="result-snippet">{result.meta.path ?? "Matched by semantic relevance."}</div>
            </div>
            <div className="result-actions">
              <button
                className="action-btn positive"
                title="Keep"
                onClick={() => void onClickRefine(result.chunk_id, "+")}
              >
                +
              </button>
              <button
                className="action-btn negative"
                title="Drop"
                onClick={() => void onClickRefine(result.chunk_id, "-")}
              >
                −
              </button>
            </div>
          </article>
        );
      })}
    </div>
  );
}

function FacetsPanel({ facets }: { facets: Facets }) {
  return (
    <div className="facets-panel">
      <div className="facets-grid">
        <FacetGroup title="Type" buckets={facets.type} />
        <FacetGroup title="Category" buckets={facets.category} />
        <FacetGroup title="Year" buckets={facets.year} />
      </div>
    </div>
  );
}

function FacetGroup({ title, buckets }: { title: string; buckets: FacetBucket[] }) {
  if (!buckets.length) return null;
  const maxTotal = Math.max(...buckets.map((b) => b.total));
  return (
    <div className="facet-group">
      <div className="facet-title">{title}</div>
      <div className="facet-items">
        {buckets.slice(0, 5).map((bucket) => (
          <div className="facet-row" key={bucket.key}>
            <span className="facet-name">{bucket.key}</span>
            <div className="facet-bar">
              <div className="facet-fill" style={{ width: `${(bucket.total / maxTotal) * 100}%` }} />
            </div>
            <span className="facet-count">{bucket.relevant}/{bucket.total}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
