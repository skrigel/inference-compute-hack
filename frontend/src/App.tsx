import { FormEvent, useEffect, useMemo, useState } from "react";
import "./App.css";
import { queryMock } from "./lib/mockAdapter";
import type { AggregateEvent, DoneEvent, QueryEvent, ResultEvent } from "./lib/types";

const DEFAULT_QUERY = "every place we retry a network call without backoff";

function App() {
  const dataMode = import.meta.env.VITE_DATA_MODE ?? "mock";
  const [predicate, setPredicate] = useState(DEFAULT_QUERY);
  const [threshold] = useState(0.5);
  const [results, setResults] = useState<ResultEvent[]>([]);
  const [aggregate, setAggregate] = useState<AggregateEvent | null>(null);
  const [done, setDone] = useState<DoneEvent | null>(null);
  const [loading, setLoading] = useState(false);

  async function runQuery(nextPredicate = predicate) {
    setLoading(true);
    setResults([]);
    setAggregate(null);
    setDone(null);

    const stream =
      dataMode === "mock"
        ? queryMock({ predicate: nextPredicate, threshold })
        : queryMock({ predicate: nextPredicate, threshold });

    for await (const event of stream) {
      applyEvent(event);
    }

    setLoading(false);
  }

  function applyEvent(event: QueryEvent) {
    if (event.type === "result") {
      setResults((current) => [...current, event]);
      return;
    }

    if (event.type === "aggregate") {
      setAggregate(event);
      return;
    }

    if (event.type === "done") {
      setDone(event);
    }
  }

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void runQuery();
  }

  useEffect(() => {
    void runQuery(DEFAULT_QUERY);
  }, []);

  const maxBin = useMemo(
    () => Math.max(1, ...(aggregate?.histogram.map((bin) => bin.count) ?? [1])),
    [aggregate],
  );

  return (
    <main className="app">
      <section className="shell">
        <header className="topbar">
          <div>
            <h1>Inference Compute Shell</h1>
            <p>Phase 0 frontend consuming contract-shaped mock stream events.</p>
          </div>
          <div className="mode">VITE_DATA_MODE={dataMode}</div>
        </header>

        <form className="query" onSubmit={onSubmit}>
          <input
            aria-label="Predicate"
            value={predicate}
            onChange={(event) => setPredicate(event.target.value)}
          />
          <button disabled={loading} type="submit">
            {loading ? "Running" : "Run query"}
          </button>
        </form>

        <section className="stats" aria-label="Stream counters">
          <div className="stat">
            <span>Results</span>
            <strong>{results.length}</strong>
          </div>
          <div className="stat">
            <span>Scanned</span>
            <strong>{aggregate?.scanned ?? done?.scanned ?? 0}</strong>
          </div>
          <div className="stat">
            <span>Matched</span>
            <strong>{aggregate?.matched ?? done?.matched ?? 0}</strong>
          </div>
          <div className="stat">
            <span>Threshold</span>
            <strong>{threshold.toFixed(2)}</strong>
          </div>
        </section>

        <section className="content">
          <div className="panel">
            <h2>Top Results</h2>
            <div className="results">
              {results.map((result) => (
                <article className="result" key={result.chunk_id}>
                  <span className="rank">#{result.rank + 1}</span>
                  <div>
                    <div className="title">{result.meta.title}</div>
                    <div className="meta">
                      {result.meta.type} / {result.meta.category ?? "uncategorized"} /{" "}
                      {result.meta.year ?? "unknown year"}
                    </div>
                  </div>
                  <span className="score">{result.score.toFixed(2)}</span>
                </article>
              ))}
            </div>
          </div>

          <aside className="panel">
            <h2>Aggregate</h2>
            <div className="histogram" aria-label="Score histogram">
              {(aggregate?.histogram ?? []).map((bin) => (
                <div
                  aria-label={`${bin.lo.toFixed(2)}-${bin.hi.toFixed(2)}: ${bin.count}`}
                  className="bar"
                  key={`${bin.lo}-${bin.hi}`}
                  style={{ height: `${Math.max(4, (bin.count / maxBin) * 100)}%` }}
                />
              ))}
            </div>
            <div className="facets">
              {(aggregate?.facets.type ?? []).map((bucket) => (
                <div className="facet" key={bucket.key}>
                  <span>{bucket.key}</span>
                  <span>
                    {bucket.relevant} / {bucket.total}
                  </span>
                </div>
              ))}
            </div>
          </aside>
        </section>

        <div className="status">{done?.summary ?? "Waiting for stream events..."}</div>
      </section>
    </main>
  );
}

export default App;
