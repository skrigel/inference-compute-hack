import {
  ChangeEvent,
  DragEvent,
  FormEvent,
  PointerEvent as ReactPointerEvent,
  ReactNode,
  useRef,
  useState,
} from "react";

import "./App.css";
import { useDashboard, type LatencyKind, type Tab } from "./hooks/useDashboard";
import type { CachedScore } from "./lib/scoreCache";
import type { Chip, FacetBucket, Facets, HistogramBin } from "./lib/types";

const DEFAULT_QUERY = "every place we retry a network call without backoff";

function App() {
  const d = useDashboard(DEFAULT_QUERY);

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void d.runQuery(d.predicate);
  };

  return (
    <div className="surface">
      <Header
        predicate={d.predicate}
        onPredicateChange={d.setPredicate}
        onSubmit={onSubmit}
        streaming={d.streaming}
        scanned={d.scanned}
        mode={d.mode}
      />

      <main className="grid">
        <section className="col left">
          <Counters
            scanned={d.scanned}
            matched={d.view.matched}
            docsPerSec={d.docsPerSec}
            latencyMs={d.latencyMs}
            latencyKind={d.latencyKind}
          />
          <Tabs active={d.activeTab} onChange={d.setActiveTab} />
          <RefinePanel
            chips={d.chips}
            refining={d.refining}
            onRefine={d.runRefine}
            onRemoveChip={d.removeChip}
            onFreshFiles={d.ingestFreshFiles}
          />

          {d.activeTab === "rel" && (
            <div className="tabpanel">
              <Histogram
                histogram={d.view.histogram}
                threshold={d.threshold}
                onThreshold={d.setThreshold}
                hasRun={d.hasRun}
              />
              <p className="eyebrow">relevant by facet · this query</p>
              <FacetBars facets={d.view.facets} hasRun={d.hasRun} />
            </div>
          )}

          {d.activeTab === "perf" && (
            <PerformanceTab
              docsPerSec={d.docsPerSec}
              elapsedMs={d.elapsedMs}
              etaMs={d.etaMs}
              latencyMs={d.latencyMs}
              latencyKind={d.latencyKind}
              latHistory={d.latHistory}
            />
          )}

          {d.activeTab === "foot" && <FootprintStub />}
        </section>

        <section className="col right">
          <p className="eyebrow feed-head">results · best-first</p>
          <ResultFeed
            results={d.view.results}
            threshold={d.threshold}
            hasRun={d.hasRun}
            onClickRefine={d.runClickRefine}
          />
        </section>
      </main>
    </div>
  );
}

interface HeaderProps {
  predicate: string;
  onPredicateChange: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  streaming: boolean;
  scanned: number;
  mode: string;
}

function Header({ predicate, onPredicateChange, onSubmit, streaming, scanned, mode }: HeaderProps) {
  return (
    <header className="topbar">
      <div className="brand">
        grep<span className="dot" />
        <b>meaning</b>
      </div>
      <form className="qwrap" onSubmit={onSubmit}>
        <input
          aria-label="Predicate"
          autoComplete="off"
          value={predicate}
          onChange={(event) => onPredicateChange(event.target.value)}
          placeholder="Describe what you're looking for — in plain English"
        />
        <button className="run" type="submit" disabled={streaming}>
          {streaming ? "scanning…" : "scan ⏎"}
        </button>
      </form>
      <div className="corpus">
        corpus: <b>{scanned.toLocaleString()}</b> scanned · <b className="amber">derived 0 B</b>
        <br />
        papers + code · recomputed live · <span className="mode">{mode}</span>
      </div>
    </header>
  );
}

interface CountersProps {
  scanned: number;
  matched: number;
  docsPerSec: number;
  latencyMs: number;
  latencyKind: LatencyKind;
}

function Counters({ scanned, matched, docsPerSec, latencyMs, latencyKind }: CountersProps) {
  return (
    <div className="counters">
      <Counter value={scanned.toLocaleString()} label="scanned" />
      <Counter value={matched.toLocaleString()} label="matched" accent />
      <Counter value={docsPerSec ? docsPerSec.toLocaleString() : "—"} label="docs / sec" />
      <Counter
        value={`${Math.round(latencyMs)}ms`}
        label={
          <>
            latency <span className={`lat-tag lat-${latencyKind}`}>{latencyKind}</span>
          </>
        }
      />
    </div>
  );
}

function Counter({ value, label, accent }: { value: string; label: ReactNode; accent?: boolean }) {
  return (
    <div className="counter">
      <div className={`cv${accent ? " accent" : ""}`}>{value}</div>
      <div className="cl">{label}</div>
    </div>
  );
}

function Tabs({ active, onChange }: { active: Tab; onChange: (tab: Tab) => void }) {
  return (
    <div className="tabs">
      <button className={`tab${active === "rel" ? " on" : ""}`} onClick={() => onChange("rel")}>
        Relevance
      </button>
      <button className="tab disabled" disabled title="deferred">
        Footprint
      </button>
      <button className={`tab${active === "perf" ? " on" : ""}`} onClick={() => onChange("perf")}>
        Performance
      </button>
    </div>
  );
}

interface RefinePanelProps {
  chips: Chip[];
  refining: boolean;
  onRefine: (utterance: string) => Promise<void>;
  onRemoveChip: (clauseId: string) => Promise<void>;
  onFreshFiles: (files: File[] | FileList) => Promise<void>;
}

function RefinePanel({ chips, refining, onRefine, onRemoveChip, onFreshFiles }: RefinePanelProps) {
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
    <section className="refine-panel">
      <form className="refine-form" onSubmit={submit}>
        <input
          aria-label="Refine"
          value={utterance}
          onChange={(event) => setUtterance(event.target.value)}
          placeholder="only python · without tests · actually papers"
        />
        <button type="submit" disabled={refining || !utterance.trim()}>
          {refining ? "…" : "refine"}
        </button>
      </form>
      <div className="chip-rail">
        {chips.map((chip) => (
          <button className="chip" key={chip.clause_id} onClick={() => void onRemoveChip(chip.clause_id)}>
            <span>{chip.label}</span>
            <b>{chip.text}</b>
            <i>×</i>
          </button>
        ))}
      </div>
      <label
        className={`dropzone${dragging ? " on" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          type="file"
          multiple
          onChange={(event: ChangeEvent<HTMLInputElement>) => ingest(event.target.files)}
        />
        fresh file
      </label>
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
    onThreshold((clientX - rect.left) / rect.width);
  };

  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!hasRun) return;
    dragging.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    setFromClientX(event.clientX);
  };

  return (
    <div className="hist">
      <div className="hist-head">
        <span>relevance distribution</span>
        <span className="thr">threshold ≥ {threshold.toFixed(2)} · drag to retune</span>
      </div>
      <div
        className="spectrum"
        ref={ref}
        onPointerDown={onPointerDown}
        onPointerMove={(event) => dragging.current && setFromClientX(event.clientX)}
        onPointerUp={() => (dragging.current = false)}
      >
        {histogram.map((bin, index) => {
          const center = (index + 0.5) / histogram.length;
          return (
            <div className={`bin${center >= threshold ? " in" : ""}`} key={index}>
              <div className="fill" style={{ height: hasRun ? `${(bin.count / max) * 100}%` : "0%" }} />
            </div>
          );
        })}
        <div className="thumb" style={{ left: `${threshold * 100}%` }} />
        <div className="axis">
          <span>0.0</span>
          <span>0.5</span>
          <span>1.0</span>
        </div>
      </div>
    </div>
  );
}

function FacetBars({ facets, hasRun }: { facets: Facets; hasRun: boolean }) {
  if (!hasRun) return null;
  return (
    <div className="facets">
      <FacetGroup title="paper vs code" buckets={facets.type} />
      <FacetGroup title="category" buckets={facets.category} />
      <FacetGroup title="year" buckets={facets.year} />
    </div>
  );
}

function FacetGroup({ title, buckets }: { title: string; buckets: FacetBucket[] }) {
  if (!buckets.length) return null;
  return (
    <div className="facet-group">
      <div className="facet-title">{title}</div>
      {buckets.map((bucket) => {
        const pct = bucket.total ? (bucket.relevant / bucket.total) * 100 : 0;
        return (
          <div className="facet-row" key={bucket.key}>
            <span className="facet-name">{bucket.key}</span>
            <div className="facet-bar">
              <div className="rel" style={{ width: `${pct}%` }} />
            </div>
            <span className="facet-count">
              {bucket.relevant}/{bucket.total}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function ResultFeed({
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
    return <div className="empty">No scan yet. Type a query and hit scan.</div>;
  }
  if (!results.length) {
    return <div className="empty">No results above the current threshold yet.</div>;
  }
  return (
    <div className="feed">
      {results.map((result) => {
        const matched = result.score >= threshold;
        return (
          <article className={`card${matched ? " matched" : ""}`} key={result.chunk_id}>
            <div className="score">
              <div className="n">{result.score.toFixed(2)}</div>
              <div className="sbar" style={{ width: `${Math.round(result.score * 100)}%` }} />
            </div>
            <div className="body">
              <div className="ctitle">{result.meta.title}</div>
              <div className="cmeta">
                <span className={`pill ${result.meta.type}`}>{result.meta.type}</span>
                <span>{result.meta.category ?? "—"}</span>
                <span>{result.meta.year ?? "—"}</span>
              </div>
              <div className="snip">{result.meta.path ?? "matched by semantic relevance to the query."}</div>
            </div>
            <div className="card-actions">
              <button title="Keep" onClick={() => void onClickRefine(result.chunk_id, "+")}>
                +
              </button>
              <button title="Drop" onClick={() => void onClickRefine(result.chunk_id, "-")}>
                -
              </button>
            </div>
          </article>
        );
      })}
    </div>
  );
}

interface PerformanceTabProps {
  docsPerSec: number;
  elapsedMs: number;
  etaMs: number;
  latencyMs: number;
  latencyKind: LatencyKind;
  latHistory: number[];
}

function PerformanceTab({ docsPerSec, elapsedMs, etaMs, latencyMs, latencyKind, latHistory }: PerformanceTabProps) {
  return (
    <div className="tabpanel">
      <div className="perf-grid">
        <PerfCard value={docsPerSec ? docsPerSec.toLocaleString() : "—"} label="docs / sec" />
        <PerfCard value={`${elapsedMs}ms`} label="wall-to-wall / query" />
        <PerfCard value={`${Math.round(latencyMs)}ms`} label={`last turn · ${latencyKind}`} accent />
        <PerfCard value={etaMs ? `${etaMs}ms` : "0ms"} label="eta (streaming)" />
      </div>
      <p className="eyebrow">latency per turn · last 16</p>
      <Sparkline values={latHistory} />
      <p className="perf-note">
        Re-thresholding re-cuts <b>cached</b> scores client-side — zero new inference. Refinement
        re-reads only the predicate suffix per chunk. As inference approaches free this flips: drop the
        cache and recompute.
      </p>
    </div>
  );
}

function PerfCard({ value, label, accent }: { value: string; label: string; accent?: boolean }) {
  return (
    <div className="perf-card">
      <div className={`pv${accent ? " accent" : ""}`}>{value}</div>
      <div className="pl">{label}</div>
    </div>
  );
}

function Sparkline({ values }: { values: number[] }) {
  if (!values.length) return <svg className="spark" viewBox="0 0 320 54" preserveAspectRatio="none" />;
  const max = Math.max(...values);
  const width = 320;
  const height = 54;
  const points = values
    .map((value, index) => {
      const x = values.length > 1 ? (index / (values.length - 1)) * width : 0;
      const y = height - 4 - (value / (max || 1)) * (height - 8);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg className="spark" viewBox="0 0 320 54" preserveAspectRatio="none">
      <polyline fill="none" stroke="#3dd9c4" strokeWidth="1.5" points={points} />
    </svg>
  );
}

function FootprintStub() {
  return (
    <div className="tabpanel">
      <div className="empty">
        <b>Footprint view — deferred.</b>
        The recompute-over-store story (derived 0 B vs RAG's growing index) lands here in a later phase.
      </div>
    </div>
  );
}

export default App;
