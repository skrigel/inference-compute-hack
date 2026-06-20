import {
  ChangeEvent,
  DragEvent,
  FormEvent,
  PointerEvent as ReactPointerEvent,
  useRef,
  useState,
} from "react";

import "./App.v2.css";
import { useDashboard, type LatencyKind } from "./hooks/useDashboard";
import type { CachedScore } from "./lib/scoreCache";
import type { Chip, FacetBucket, Facets, HistogramBin } from "./lib/types";

type MainTab = "results" | "facets" | "analytics";

function App() {
  const d = useDashboard();
  const [activeTab, setActiveTab] = useState<MainTab>("results");

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void d.runQuery(d.predicate);
  };

  return (
    <div className="app-wrapper">
      <HeaderBar />
      <main className="main-content">
        <QueryBar
          predicate={d.predicate}
          onPredicateChange={d.setPredicate}
          onSubmit={onSubmit}
          streaming={d.streaming}
        />
        <ThresholdControl
          histogram={d.view.histogram}
          threshold={d.threshold}
          onThreshold={d.setThreshold}
          hasRun={d.hasRun}
          matched={d.view.matched}
          scanned={d.scanned}
        />
        <FilterBar
          chips={d.chips}
          refining={d.refining}
          onRefine={d.runRefine}
          onRemoveChip={d.removeChip}
          onFreshFiles={d.ingestFreshFiles}
        />
        <TabbedContent
          activeTab={activeTab}
          onTabChange={setActiveTab}
          results={d.view.results}
          threshold={d.threshold}
          hasRun={d.hasRun}
          streaming={d.streaming}
          onClickRefine={d.runClickRefine}
          facets={d.view.facets}
          docsPerSec={d.docsPerSec}
          elapsedMs={d.elapsedMs}
          etaMs={d.etaMs}
          latencyMs={d.latencyMs}
          latencyKind={d.latencyKind}
          latHistory={d.latHistory}
        />
      </main>
    </div>
  );
}

function HeaderBar() {
  return (
    <header className="header-bar">
      <div className="header-brand">grep<span>meaning</span></div>
      <div className="header-spacer" />
      <button className="header-icon-btn" aria-label="Help">
        <svg viewBox="0 0 20 20" fill="currentColor" width="18" height="18">
          <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM8.94 6.94a.75.75 0 11-1.061-1.061 3 3 0 112.871 5.026v.345a.75.75 0 01-1.5 0v-.5c0-.72.57-1.172 1.081-1.287A1.5 1.5 0 108.94 6.94zM10 15a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
        </svg>
      </button>
      <button className="header-icon-btn" aria-label="Settings">
        <svg viewBox="0 0 20 20" fill="currentColor" width="18" height="18">
          <path fillRule="evenodd" d="M7.84 1.804A1 1 0 018.82 1h2.36a1 1 0 01.98.804l.331 1.652a6.993 6.993 0 011.929 1.115l1.598-.54a1 1 0 011.186.447l1.18 2.044a1 1 0 01-.205 1.251l-1.267 1.113a7.047 7.047 0 010 2.228l1.267 1.113a1 1 0 01.206 1.25l-1.18 2.045a1 1 0 01-1.187.447l-1.598-.54a6.993 6.993 0 01-1.929 1.115l-.33 1.652a1 1 0 01-.98.804H8.82a1 1 0 01-.98-.804l-.331-1.652a6.993 6.993 0 01-1.929-1.115l-1.598.54a1 1 0 01-1.186-.447l-1.18-2.044a1 1 0 01.205-1.251l1.267-1.114a7.05 7.05 0 010-2.227L1.821 7.773a1 1 0 01-.206-1.25l1.18-2.045a1 1 0 011.187-.447l1.598.54A6.993 6.993 0 017.51 3.456l.33-1.652zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
        </svg>
      </button>
    </header>
  );
}

interface QueryBarProps {
  predicate: string;
  onPredicateChange: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  streaming: boolean;
}

function QueryBar({ predicate, onPredicateChange, onSubmit, streaming }: QueryBarProps) {
  return (
    <section className="query-section">
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

interface ThresholdControlProps {
  histogram: HistogramBin[];
  threshold: number;
  onThreshold: (value: number) => void;
  hasRun: boolean;
  matched: number;
  scanned: number;
}

function ThresholdControl({ histogram, threshold, onThreshold, hasRun, matched, scanned }: ThresholdControlProps) {
  const ref = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const max = Math.max(1, ...histogram.map((bin) => bin.count));

  const setFromClientX = (clientX: number) => {
    const element = ref.current;
    if (!element) return;
    const rect = element.getBoundingClientRect();
    const raw = (clientX - rect.left) / rect.width;
    onThreshold(Math.max(0, Math.min(1, raw)));
  };

  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!hasRun) return;
    dragging.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    setFromClientX(event.clientX);
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (!hasRun) return;
    const step = event.shiftKey ? 0.1 : 0.01;
    if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
      event.preventDefault();
      onThreshold(Math.max(0, threshold - step));
    } else if (event.key === "ArrowRight" || event.key === "ArrowUp") {
      event.preventDefault();
      onThreshold(Math.min(1, threshold + step));
    }
  };

  return (
    <section className="threshold-section">
      <div className="threshold-header">
        <span className="threshold-label">Threshold</span>
        <span className="threshold-stats">
          <strong>{matched.toLocaleString()}</strong> of {scanned.toLocaleString()} matched
          {hasRun && <span className="threshold-value">≥ {threshold.toFixed(2)}</span>}
        </span>
      </div>
      <div
        className={`histogram${hasRun ? "" : " empty"}`}
        ref={ref}
        role="slider"
        tabIndex={hasRun ? 0 : -1}
        aria-label="Score threshold"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={threshold}
        aria-valuetext={`${threshold.toFixed(2)} threshold, ${matched} results`}
        onPointerDown={onPointerDown}
        onPointerMove={(event) => dragging.current && setFromClientX(event.clientX)}
        onPointerUp={() => (dragging.current = false)}
        onKeyDown={onKeyDown}
      >
        <div className="histogram-bars">
          {histogram.map((bin, index) => {
            const center = (index + 0.5) / histogram.length;
            return (
              <div className={`bin${center >= threshold ? " in" : ""}`} key={index}>
                <div
                  className="fill"
                  style={{ height: hasRun ? `${(bin.count / max) * 100}%` : "0%" }}
                />
              </div>
            );
          })}
        </div>
        <div className="threshold-thumb" style={{ left: `${threshold * 100}%` }} />
        <div className="histogram-axis">
          <span>0</span>
          <span>1</span>
        </div>
      </div>
      {!hasRun && (
        <div className="threshold-hint">Enter a query and click Scan to see score distribution</div>
      )}
    </section>
  );
}

interface FilterBarProps {
  chips: Chip[];
  refining: boolean;
  onRefine: (utterance: string) => Promise<void>;
  onRemoveChip: (clauseId: string) => Promise<void>;
  onFreshFiles: (files: File[] | FileList) => Promise<void>;
}

function FilterBar({ chips, refining, onRefine, onRemoveChip, onFreshFiles }: FilterBarProps) {
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
            <button className="chip" key={chip.clause_id} onClick={() => void onRemoveChip(chip.clause_id)} aria-label={`Remove filter: ${chip.text}`}>
              <span className="chip-label">{chip.label}</span>
              <span className="chip-text">{chip.text}</span>
              <span className="chip-remove" aria-hidden="true">×</span>
            </button>
          ))}
          <form className="add-filter-form" onSubmit={submit}>
            <input
              aria-label="Add filter"
              value={utterance}
              onChange={(event) => setUtterance(event.target.value)}
              placeholder="Refine: only python, without tests..."
              disabled={refining}
            />
            {utterance.trim() && (
              <button className="btn-add" type="submit" disabled={refining}>{refining ? "..." : "Add"}</button>
            )}
          </form>
          <label
            className={`dropzone${dragging ? " active" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <input type="file" multiple onChange={(event: ChangeEvent<HTMLInputElement>) => ingest(event.target.files)} />
            <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
              <path d="M10 3a.75.75 0 01.75.75v10.638l3.96-4.158a.75.75 0 111.08 1.04l-5.25 5.5a.75.75 0 01-1.08 0l-5.25-5.5a.75.75 0 111.08-1.04l3.96 4.158V3.75A.75.75 0 0110 3z" />
            </svg>
            <span>Files</span>
          </label>
        </div>
      </div>
    </section>
  );
}

function FacetGroup({ title, buckets }: { title: string; buckets: FacetBucket[] }) {
  if (!buckets.length) return null;
  return (
    <div className="facet-group">
      <div className="facet-title">{title}</div>
      <div className="facet-items">
        {buckets.map((bucket) => {
          const pct = bucket.total ? (bucket.relevant / bucket.total) * 100 : 0;
          return (
            <div className="facet-row" key={bucket.key}>
              <span className="facet-name">{bucket.key}</span>
              <div className="facet-bar">
                <div className="facet-fill" style={{ width: `${pct}%` }} />
              </div>
              <span className="facet-count">{bucket.relevant}/{bucket.total}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

interface TabbedContentProps {
  activeTab: MainTab;
  onTabChange: (tab: MainTab) => void;
  results: CachedScore[];
  threshold: number;
  hasRun: boolean;
  streaming: boolean;
  onClickRefine: (chunkId: string, sign: "+" | "-") => Promise<void>;
  facets: Facets;
  docsPerSec: number;
  elapsedMs: number;
  etaMs: number;
  latencyMs: number;
  latencyKind: LatencyKind;
  latHistory: number[];
}

function TabbedContent({
  activeTab, onTabChange, results, threshold, hasRun, streaming, onClickRefine,
  facets, docsPerSec, elapsedMs, etaMs, latencyMs, latencyKind, latHistory,
}: TabbedContentProps) {
  const hasFacets = facets.type.length > 0 || facets.category.length > 0;
  return (
    <section className="tabbed-section">
      <div className="tab-bar">
        <button className={`tab-btn${activeTab === "results" ? " active" : ""}`} onClick={() => onTabChange("results")}>
          Results {hasRun && <span className="tab-count">{results.length}</span>}
        </button>
        <button className={`tab-btn${activeTab === "facets" ? " active" : ""}`} onClick={() => onTabChange("facets")}>
          Breakdown
        </button>
        <button className={`tab-btn${activeTab === "analytics" ? " active" : ""}`} onClick={() => onTabChange("analytics")}>
          Performance
        </button>
        {hasRun && (
          <div className="tab-bar-metrics">
            {streaming && etaMs > 0 && <span className="eta-tag">ETA {Math.round(etaMs / 1000)}s</span>}
            {!streaming && latencyMs > 0 && (
              <span className={`latency-tag lat-${latencyKind}`}>{Math.round(latencyMs)}ms <span className="lat-kind">{latencyKind}</span></span>
            )}
          </div>
        )}
      </div>
      {activeTab === "results" && (
        <ResultsPanel results={results} threshold={threshold} hasRun={hasRun} streaming={streaming} onClickRefine={onClickRefine} />
      )}
      {activeTab === "facets" && (
        <div className="tab-panel facets-panel">
          {!hasRun ? (
            <div className="panel-empty"><p>Run a query to see breakdown by type and category</p></div>
          ) : !hasFacets ? (
            <div className="panel-empty"><p>No facet data available</p></div>
          ) : (
            <div className="facets-grid">
              <FacetGroup title="Type" buckets={facets.type} />
              <FacetGroup title="Category" buckets={facets.category} />
            </div>
          )}
        </div>
      )}
      {activeTab === "analytics" && (
        <AnalyticsPanel docsPerSec={docsPerSec} elapsedMs={elapsedMs} etaMs={etaMs} latencyMs={latencyMs} latencyKind={latencyKind} latHistory={latHistory} />
      )}
    </section>
  );
}

function ResultsPanel({
  results, threshold, hasRun, streaming, onClickRefine,
}: {
  results: CachedScore[];
  threshold: number;
  hasRun: boolean;
  streaming: boolean;
  onClickRefine: (chunkId: string, sign: "+" | "-") => Promise<void>;
}) {
  const [selected, setSelected] = useState<CachedScore | null>(null);

  if (!hasRun) {
    return (
      <div className="tab-panel">
        <div className="panel-empty">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="40" height="40">
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
          </svg>
          <h3>Start a search</h3>
          <p>Describe what you're looking for in natural language</p>
        </div>
      </div>
    );
  }
  if (streaming) {
    return (
      <div className="tab-panel">
        {[...Array(5)].map((_, i) => (
          <div className="result-skeleton" key={i}>
            <div className="skeleton-score" />
            <div className="skeleton-body">
              <div className="skeleton-title" />
              <div className="skeleton-meta" />
            </div>
          </div>
        ))}
      </div>
    );
  }
  if (!results.length) {
    return (
      <div className="tab-panel">
        <div className="panel-empty">
          <h3>No matches</h3>
          <p>Try lowering the threshold or adjusting your query</p>
        </div>
      </div>
    );
  }
  return (
    <div className={`tab-panel results-split${selected ? " has-detail" : ""}`}>
      <div className="results-list">
        {results.map((result) => {
          const matched = result.score >= threshold;
          const typeLabel = result.meta.type === "code" ? "Code" : "Paper";
          const isSelected = selected?.chunk_id === result.chunk_id;
          return (
            <article
              className={`result-row${matched ? " matched" : ""}${isSelected ? " selected" : ""}`}
              key={result.chunk_id}
              onClick={() => setSelected(isSelected ? null : result)}
            >
              <div className="result-score">
                <span className="score-value">{result.score.toFixed(2)}</span>
                <div className="score-bar">
                  <div className="score-fill" style={{ width: `${Math.round(result.score * 100)}%` }} />
                </div>
              </div>
              <div className="result-content">
                <div className="result-header">
                  <span className={`type-badge type-${result.meta.type ?? "paper"}`}>{typeLabel}</span>
                  <span className="result-title">{result.meta.title}</span>
                </div>
                <div className="result-meta">{result.meta.category ?? "—"} {result.meta.path && `· ${result.meta.path}`}</div>
              </div>
              <div className="result-actions" onClick={(e) => e.stopPropagation()}>
                <button className="action-btn positive" onClick={() => void onClickRefine(result.chunk_id, "+")} aria-label="More like this" title="More like this">+</button>
                <button className="action-btn negative" onClick={() => void onClickRefine(result.chunk_id, "-")} aria-label="Less like this" title="Less like this">−</button>
              </div>
            </article>
          );
        })}
      </div>
      {selected && (
        <aside className="result-detail">
          <div className="detail-header">
            <span className="detail-title">{selected.meta.title}</span>
            <button className="detail-close" onClick={() => setSelected(null)} aria-label="Close">×</button>
          </div>
          <div className="detail-body">
            <div className="detail-row">
              <span className="detail-label">Score</span>
              <span className="detail-value score-accent">{selected.score.toFixed(3)}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Type</span>
              <span className="detail-value">{selected.meta.type === "code" ? "Code" : "Paper"}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Category</span>
              <span className="detail-value">{selected.meta.category ?? "—"}</span>
            </div>
            {selected.meta.path && (
              <div className="detail-row">
                <span className="detail-label">Path</span>
                <span className="detail-value mono">{selected.meta.path}</span>
              </div>
            )}
            <div className="detail-section">
              <span className="detail-label">Why it matched</span>
              <p className="detail-rationale">
                {selected.meta.type === "code"
                  ? `This ${selected.meta.lang ?? "code"} file in ${selected.meta.repo ?? "the repository"} contains patterns related to your query.`
                  : `This paper from ${selected.meta.category ?? "the archive"} discusses concepts semantically similar to your predicate.`}
              </p>
            </div>
          </div>
          <div className="detail-actions">
            <button className="btn-detail positive" onClick={() => void onClickRefine(selected.chunk_id, "+")}>More like this</button>
            <button className="btn-detail negative" onClick={() => void onClickRefine(selected.chunk_id, "-")}>Less like this</button>
          </div>
        </aside>
      )}
    </div>
  );
}

function AnalyticsPanel({
  docsPerSec, elapsedMs, etaMs, latencyMs, latencyKind, latHistory,
}: {
  docsPerSec: number;
  elapsedMs: number;
  etaMs: number;
  latencyMs: number;
  latencyKind: LatencyKind;
  latHistory: number[];
}) {
  return (
    <div className="tab-panel analytics-panel">
      <div className="analytics-grid">
        <div className="stat-card">
          <div className="stat-value">{docsPerSec ? docsPerSec.toLocaleString() : "—"}</div>
          <div className="stat-label">Docs/sec</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{elapsedMs}ms</div>
          <div className="stat-label">Wall time</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{Math.round(latencyMs)}ms</div>
          <div className="stat-label">Latency <span className={`lat-tag lat-${latencyKind}`}>{latencyKind}</span></div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{etaMs ? `${etaMs}ms` : "—"}</div>
          <div className="stat-label">ETA</div>
        </div>
      </div>
      <div className="sparkline-section">
        <div className="section-label">Latency history</div>
        <div className="sparkline-container">
          <Sparkline values={latHistory} />
        </div>
      </div>
      <div className="analytics-note">
        Re-thresholding uses cached scores client-side. Refinement re-reads only the predicate suffix per chunk.
      </div>
    </div>
  );
}

function Sparkline({ values }: { values: number[] }) {
  if (!values.length) {
    return <svg className="sparkline" viewBox="0 0 320 48" />;
  }

  const max = Math.max(...values);
  const width = 320;
  const height = 48;
  const pad = 2;

  const points = values.map((v, i) => {
    const x = values.length > 1 ? pad + (i / (values.length - 1)) * (width - 2 * pad) : width / 2;
    const y = height - pad - (v / (max || 1)) * (height - 2 * pad);
    return { x, y };
  });

  const line = points.map((p) => `${p.x},${p.y}`).join(" ");
  const area = [
    `${points[0].x},${height - pad}`,
    ...points.map((p) => `${p.x},${p.y}`),
    `${points[points.length - 1].x},${height - pad}`,
  ].join(" ");

  return (
    <svg className="sparkline" viewBox={`0 0 ${width} ${height}`}>
      <polygon className="sparkline-fill" points={area} />
      <polyline className="sparkline-line" points={line} />
      {points.length > 0 && (
        <circle className="sparkline-dot" cx={points[points.length - 1].x} cy={points[points.length - 1].y} r="3" />
      )}
    </svg>
  );
}

export default App;
