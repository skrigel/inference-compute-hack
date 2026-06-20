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
import { DocumentPreview } from "../components/DocumentPreview";

export function SearchPage() {
  const { corpusId } = useParams<{ corpusId: string }>();
  const navigate = useNavigate();
  const [corpus, setCorpus] = useState<Corpus | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeCorpus, setActiveCorpus] = useState<"demo" | "browsecomp">("demo");
  const [switchingCorpus, setSwitchingCorpus] = useState(false);
  const [previewResult, setPreviewResult] = useState<CachedScore | null>(null);
  const d = useDashboard();

  const handleCorpusChange = async (newCorpus: "demo" | "browsecomp") => {
    if (newCorpus === activeCorpus || switchingCorpus) return;
    setSwitchingCorpus(true);
    try {
      // For browsecomp, limit to 1000 docs by default for faster loading
      const limit = newCorpus === "browsecomp" ? 1000 : undefined;
      await d.ingestCorpus(newCorpus, limit);
      setActiveCorpus(newCorpus);
      await d.runQuery(d.predicate);
    } finally {
      setSwitchingCorpus(false);
    }
  };

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
        activeCorpus={activeCorpus}
        switchingCorpus={switchingCorpus}
        onCorpusChange={handleCorpusChange}
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

      <ComputePanel d={d} />

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
        docsPerSec={d.docsPerSec}
        elapsedMs={d.elapsedMs}
        latHistory={d.latHistory}
        selectedIds={d.selection?.selectedIds ?? []}
        onClickRefine={d.runClickRefine}
        onResultClick={(result) => setPreviewResult(result)}
      />

      {previewResult && (
        <DocumentPreview result={previewResult} onClose={() => setPreviewResult(null)} />
      )}
    </main>
  );
}

interface QuerySectionProps {
  predicate: string;
  onPredicateChange: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  streaming: boolean;
  corpus: Corpus | null;
  activeCorpus: "demo" | "browsecomp";
  switchingCorpus: boolean;
  onCorpusChange: (corpus: "demo" | "browsecomp") => void;
}

function QuerySection({ predicate, onPredicateChange, onSubmit, streaming, corpus, activeCorpus, switchingCorpus, onCorpusChange }: QuerySectionProps) {
  return (
    <section className="query-section">
      <div className="query-header">
        <span className="corpus-name">{corpus?.name ?? "Unknown Corpus"}</span>
        <span className="corpus-docs">{corpus?.documentCount ?? 0} documents</span>
        <select
          className="corpus-select"
          value={activeCorpus}
          onChange={(e) => onCorpusChange(e.target.value as "demo" | "browsecomp")}
          disabled={switchingCorpus || streaming}
        >
          <option value="demo">Demo (7 docs)</option>
          <option value="browsecomp">BrowseComp+ (1k docs)</option>
        </select>
        {switchingCorpus && <span className="corpus-loading">Loading...</span>}
      </div>
      <form className="query-form" onSubmit={onSubmit}>
        <input
          aria-label="Search query"
          autoComplete="off"
          value={predicate}
          onChange={(event) => onPredicateChange(event.target.value)}
          placeholder="Type a query..."
        />
        <button className="btn-primary" type="submit" disabled={streaming || switchingCorpus}>
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

interface ComputePanelProps {
  d: ReturnType<typeof useDashboard>;
}

// The Compute panel hosts the three infinite-compute axes as explicit dials:
// Memory (corpus in scope), Movement (what to move), Truth (predicate beam).
function ComputePanel({ d }: ComputePanelProps) {
  return (
    <section className="compute-panel">
      <div className="compute-header">
        <span className="compute-title">Compute</span>
        <span className="compute-sub">one budget · three axes</span>
      </div>
      <div className="compute-grid">
        <div className="axis-card">
          <div className="axis-head">
            <span className="axis-tag axis-memory">Memory</span>
            <span className="axis-name">Corpus in scope</span>
          </div>
          <input
            className="axis-range"
            type="range"
            min={5}
            max={100}
            step={5}
            value={Math.round(d.computeBudget * 100)}
            onChange={(event) => d.setComputeBudget(Number(event.target.value) / 100)}
            onPointerUp={() => d.rescan()}
            onKeyUp={() => d.rescan()}
            aria-label="Compute budget"
          />
          <div className="axis-readout">
            <strong>{Math.round(d.computeBudget * 100)}%</strong> budget · scored{" "}
            <strong>{d.corpusScope.scored}</strong> of {d.corpusScope.total} chunks
          </div>
        </div>

        <div className="axis-card">
          <div className="axis-head">
            <span className="axis-tag axis-movement">Movement</span>
            <span className="axis-name">What to move</span>
          </div>
          <label className="axis-control">
            <span>Precision target {d.precisionTarget.toFixed(2)}</span>
            <input
              className="axis-range"
              type="range"
              min={50}
              max={99}
              step={1}
              value={Math.round(d.precisionTarget * 100)}
              onChange={(event) => d.setPrecisionTarget(Number(event.target.value) / 100)}
            />
          </label>
          <div className="axis-control-row">
            <label>
              Move K
              <input
                type="number"
                min={1}
                max={20}
                value={d.movementBudget}
                onChange={(event) => d.setMovementBudget(Number(event.target.value))}
              />
            </label>
            <label>
              Beam B
              <input
                type="number"
                min={1}
                max={16}
                value={d.selectionBeamWidth}
                onChange={(event) => d.setSelectionBeamWidth(Number(event.target.value))}
              />
            </label>
          </div>
          <div className="axis-buttons">
            <button className="axis-btn" onClick={() => d.autoThreshold()} disabled={!d.hasRun}>
              Auto-threshold
            </button>
            <button className="axis-btn" onClick={() => d.smartSelect()} disabled={!d.hasRun}>
              Smart select
            </button>
            {d.selection && (
              <button className="axis-btn ghost" onClick={() => d.clearSelection()}>
                Clear
              </button>
            )}
          </div>
          {d.selection && (
            <div className="axis-readout">
              {d.selection.mode === "threshold" ? (
                <>
                  threshold <strong>{d.selection.threshold.toFixed(2)}</strong> ·{" "}
                  <strong>{d.selection.selectedIds.length}</strong> selected
                </>
              ) : (
                <>
                  <strong>{d.selection.selectedIds.length}</strong> moved · {d.selection.coveredFacets.length}{" "}
                  facets · obj <strong>{d.selection.objective.toFixed(2)}</strong> (greedy floor{" "}
                  {d.selection.greedyObjective.toFixed(2)})
                </>
              )}
            </div>
          )}
        </div>

        <div className="axis-card">
          <div className="axis-head">
            <span className="axis-tag axis-truth">Truth</span>
            <span className="axis-name">Predicate beam</span>
          </div>
          <label className="axis-control">
            <span>
              Beam width {d.beamWidth}
              {d.beamWidth === 1 ? " · single clause" : " · search"}
            </span>
            <input
              className="axis-range"
              type="range"
              min={1}
              max={8}
              step={1}
              value={d.beamWidth}
              onChange={(event) => d.setBeamWidth(Number(event.target.value))}
            />
          </label>
          <div className="axis-readout axis-hint">
            {d.beamWidth === 1
              ? "Refine tries one clause (human / agent drives)."
              : `Refine explores ${d.beamWidth} candidates; the objective keeps the best.`}
          </div>
          {d.beamCandidates && d.beamCandidates.length > 0 && (
            <div className="beam-candidates">
              {d.beamCandidates.map((candidate, index) => (
                <div className={`beam-row${candidate.chosen ? " chosen" : ""}`} key={index}>
                  <span className="beam-text">{candidate.text}</span>
                  <span className="beam-metric">
                    obj {candidate.objective.toFixed(2)} · cov {Math.round(candidate.coverage * 100)}%
                  </span>
                </div>
              ))}
            </div>
          )}
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
  docsPerSec: number;
  elapsedMs: number;
  latHistory: number[];
  selectedIds: string[];
  onClickRefine: (chunkId: string, sign: "+" | "-") => Promise<void>;
  onResultClick: (result: CachedScore) => void;
}

function TabbedSection({
  hasRun,
  results,
  facets,
  threshold,
  matched,
  etaMs,
  latencyMs,
  latencyKind,
  docsPerSec,
  elapsedMs,
  latHistory,
  selectedIds,
  onClickRefine,
  onResultClick,
}: TabbedSectionProps) {
  const [tab, setTab] = useState<"results" | "analytics" | "facets">("results");

  return (
    <section className="tabbed-section">
      <div className="tab-bar">
        <button className={`tab-btn${tab === "results" ? " active" : ""}`} onClick={() => setTab("results")}>
          Results <span className="tab-count">{matched}</span>
        </button>
        <button className={`tab-btn${tab === "analytics" ? " active" : ""}`} onClick={() => setTab("analytics")}>
          Analytics
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
          <ResultList
            results={results}
            threshold={threshold}
            hasRun={hasRun}
            selectedIds={selectedIds}
            onClickRefine={onClickRefine}
            onResultClick={onResultClick}
          />
        )}
        {tab === "analytics" && (
          <AnalyticsPanel
            docsPerSec={docsPerSec}
            elapsedMs={elapsedMs}
            latencyMs={latencyMs}
            latencyKind={latencyKind}
            etaMs={etaMs}
            latHistory={latHistory}
          />
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
  selectedIds,
  onClickRefine,
  onResultClick,
}: {
  results: CachedScore[];
  threshold: number;
  hasRun: boolean;
  selectedIds: string[];
  onClickRefine: (chunkId: string, sign: "+" | "-") => Promise<void>;
  onResultClick: (result: CachedScore) => void;
}) {
  const selected = new Set(selectedIds);
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
        const isSelected = selected.has(result.chunk_id);
        return (
          <article
            className={`result-row${matched ? " matched" : ""}${isSelected ? " selected" : ""}`}
            key={result.chunk_id}
            onClick={() => onResultClick(result)}
          >
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
                onClick={(e) => { e.stopPropagation(); void onClickRefine(result.chunk_id, "+"); }}
              >
                +
              </button>
              <button
                className="action-btn negative"
                title="Drop"
                onClick={(e) => { e.stopPropagation(); void onClickRefine(result.chunk_id, "-"); }}
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

interface AnalyticsPanelProps {
  docsPerSec: number;
  elapsedMs: number;
  latencyMs: number;
  latencyKind: LatencyKind;
  etaMs: number;
  latHistory: number[];
}

function AnalyticsPanel({ docsPerSec, elapsedMs, latencyMs, latencyKind, etaMs, latHistory }: AnalyticsPanelProps) {
  return (
    <div className="analytics-panel">
      <div className="analytics-grid">
        <div className="stat-card">
          <div className="stat-value">{docsPerSec ? docsPerSec.toLocaleString() : "—"}</div>
          <div className="stat-label">Docs / sec</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{elapsedMs}ms</div>
          <div className="stat-label">Wall time</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{Math.round(latencyMs)}ms</div>
          <div className="stat-label">
            Latency <span className={`lat-tag lat-${latencyKind}`}>{latencyKind}</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{etaMs ? `${etaMs}ms` : "—"}</div>
          <div className="stat-label">ETA</div>
        </div>
      </div>

      <div className="sparkline-section">
        <div className="section-label">Latency History</div>
        <div className="sparkline-container">
          <Sparkline values={latHistory} />
        </div>
      </div>

      <div className="analytics-note">
        Re-thresholding uses cached scores client-side (zero inference). Refinement re-reads only the predicate suffix per chunk.
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
