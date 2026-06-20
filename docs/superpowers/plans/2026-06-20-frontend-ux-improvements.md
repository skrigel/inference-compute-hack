# Frontend UX Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add settings toggle for mock/live mode, built-in corpora section on homepage, remove auto-execute query, and document preview on result click.

**Architecture:** React components with localStorage for settings persistence. No global state store - local component state and hooks only. Modal-based document preview with two-step interaction (metadata card → full preview).

**Tech Stack:** React 19, TypeScript, React Router, localStorage

## Global Constraints

- TypeScript strict mode
- Named exports only (no default exports except App.tsx)
- Follow existing code patterns in the codebase
- No new dependencies

---

## File Structure

```
src/
├── hooks/
│   └── useSettings.ts (create)     # Settings hook for localStorage
├── pages/
│   ├── HomePage.tsx (modify)       # Add built-in corpora section
│   ├── SearchPage.tsx (modify)     # Remove auto-query, add preview click
│   └── SettingsPage.tsx (create)   # Settings page with API mode toggle
├── components/
│   ├── NavBar.tsx (modify)         # Link settings button to /settings
│   └── DocumentPreview.tsx (create) # Metadata card + full preview modal
├── lib/
│   └── api.ts (modify)             # Read mode from localStorage, support switching
└── App.tsx (modify)                # Add /settings route
```

---

### Task 1: Settings Hook and API Mode Switching

**Files:**
- Create: `src/hooks/useSettings.ts`
- Modify: `src/lib/api.ts`
- Test: `src/__tests__/useSettings.test.ts`

**Interfaces:**
- Produces: `useSettings()` hook returning `{ apiMode, setApiMode }` where `apiMode: "mock" | "live"`
- Produces: `getApiMode(): DataMode` function in api.ts
- Produces: `setApiMode(mode: DataMode): void` function in api.ts

- [ ] **Step 1: Write failing test for useSettings hook**

Create `src/__tests__/useSettings.test.ts`:

```typescript
import { renderHook, act } from "@testing-library/react";
import { useSettings } from "../hooks/useSettings";

describe("useSettings", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("defaults to mock mode", () => {
    const { result } = renderHook(() => useSettings());
    expect(result.current.apiMode).toBe("mock");
  });

  it("persists mode to localStorage", () => {
    const { result } = renderHook(() => useSettings());
    act(() => {
      result.current.setApiMode("live");
    });
    expect(result.current.apiMode).toBe("live");
    expect(localStorage.getItem("api-mode")).toBe("live");
  });

  it("reads initial value from localStorage", () => {
    localStorage.setItem("api-mode", "live");
    const { result } = renderHook(() => useSettings());
    expect(result.current.apiMode).toBe("live");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- --run src/__tests__/useSettings.test.ts`
Expected: FAIL with "Cannot find module '../hooks/useSettings'"

- [ ] **Step 3: Write useSettings hook**

Create `src/hooks/useSettings.ts`:

```typescript
import { useCallback, useSyncExternalStore } from "react";

export type ApiMode = "mock" | "live";

const STORAGE_KEY = "api-mode";

function getSnapshot(): ApiMode {
  const stored = localStorage.getItem(STORAGE_KEY);
  return stored === "live" ? "live" : "mock";
}

function subscribe(callback: () => void): () => void {
  const handler = (event: StorageEvent) => {
    if (event.key === STORAGE_KEY) callback();
  };
  window.addEventListener("storage", handler);
  return () => window.removeEventListener("storage", handler);
}

export function useSettings() {
  const apiMode = useSyncExternalStore(subscribe, getSnapshot, () => "mock" as ApiMode);

  const setApiMode = useCallback((mode: ApiMode) => {
    localStorage.setItem(STORAGE_KEY, mode);
    window.dispatchEvent(new StorageEvent("storage", { key: STORAGE_KEY, newValue: mode }));
  }, []);

  return { apiMode, setApiMode };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test -- --run src/__tests__/useSettings.test.ts`
Expected: PASS

- [ ] **Step 5: Modify api.ts to support runtime mode switching**

Edit `src/lib/api.ts`. Replace the static MODE constant and createApi with dynamic mode reading:

```typescript
import { deleteClauseLive, ingestLive, queryLive, refineLive } from "./liveAdapter";
import { deleteClauseMock, ingestMock, queryMock, refineMock } from "./mockAdapter";
import type { Facets, FreshDocument, QueryEvent, QueryRequest, RefineEvent, RefineRequest } from "./types";

export type DataMode = "mock" | "live";

const STORAGE_KEY = "api-mode";

export function getApiMode(): DataMode {
  if (typeof window === "undefined") return "mock";
  const stored = localStorage.getItem(STORAGE_KEY);
  return stored === "live" ? "live" : "mock";
}

export function setApiMode(mode: DataMode): void {
  localStorage.setItem(STORAGE_KEY, mode);
  window.dispatchEvent(new StorageEvent("storage", { key: STORAGE_KEY, newValue: mode }));
}

export interface DashboardApi {
  mode: DataMode;
  ingest(corpusId: string, documents?: FreshDocument[], limit?: number): Promise<{ n_chunks: number; facets: Facets }>;
  query(request: QueryRequest, onEvent: (event: QueryEvent) => void, signal?: AbortSignal): Promise<void>;
  refine(request: RefineRequest, onEvent: (event: RefineEvent) => void, signal?: AbortSignal): Promise<void>;
  deleteClause(clauseId: string): Promise<{ removed: boolean; refine_ms: number }>;
}

async function queryViaMock(
  request: QueryRequest,
  onEvent: (event: QueryEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  for await (const event of queryMock(request, signal)) {
    if (signal?.aborted) break;
    onEvent(event);
  }
}

async function refineViaMock(
  request: RefineRequest,
  onEvent: (event: RefineEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  for await (const event of refineMock(request, signal)) {
    if (signal?.aborted) break;
    onEvent(event);
  }
}

function isAbort(error: unknown): boolean {
  return typeof error === "object" && error !== null && (error as { name?: string }).name === "AbortError";
}

export function createApi(): DashboardApi {
  return {
    get mode(): DataMode {
      return getApiMode();
    },
    async ingest(corpusId, documents, limit) {
      if (getApiMode() === "live") {
        try {
          return await ingestLive(corpusId, documents, limit);
        } catch (error) {
          console.warn("live ingest failed; falling back to mock", error);
          return ingestMock(corpusId, documents);
        }
      }
      return ingestMock(corpusId, documents);
    },
    async query(request, onEvent, signal) {
      if (getApiMode() === "live") {
        try {
          await queryLive(request, onEvent, signal);
          return;
        } catch (error) {
          if (isAbort(error)) return;
          console.warn("live query failed; falling back to mock", error);
        }
      }
      await queryViaMock(request, onEvent, signal);
    },
    async refine(request, onEvent, signal) {
      if (getApiMode() === "live") {
        try {
          await refineLive(request, onEvent, signal);
          return;
        } catch (error) {
          if (isAbort(error)) return;
          console.warn("live refine failed; falling back to mock", error);
        }
      }
      await refineViaMock(request, onEvent, signal);
    },
    async deleteClause(clauseId) {
      if (getApiMode() === "live") {
        try {
          return await deleteClauseLive(clauseId);
        } catch (error) {
          console.warn("live delete clause failed; falling back to mock", error);
          return deleteClauseMock(clauseId);
        }
      }
      return deleteClauseMock(clauseId);
    },
  };
}

export const api = createApi();
```

- [ ] **Step 6: Run all tests to verify no regressions**

Run: `cd frontend && npm test -- --run`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add frontend/src/hooks/useSettings.ts frontend/src/lib/api.ts frontend/src/__tests__/useSettings.test.ts
git commit -m "feat: add useSettings hook and runtime API mode switching"
```

---

### Task 2: Settings Page

**Files:**
- Create: `src/pages/SettingsPage.tsx`
- Modify: `src/App.tsx`
- Modify: `src/components/NavBar.tsx`

**Interfaces:**
- Consumes: `useSettings()` from `../hooks/useSettings`
- Produces: `SettingsPage` component at route `/settings`

- [ ] **Step 1: Create SettingsPage component**

Create `src/pages/SettingsPage.tsx`:

```typescript
import { useSettings } from "../hooks/useSettings";

export function SettingsPage() {
  const { apiMode, setApiMode } = useSettings();

  return (
    <main className="page-content">
      <div className="page-header">
        <h1 className="page-title">Settings</h1>
      </div>

      <section className="settings-section">
        <h2 className="section-title">Developer Options</h2>

        <label className="setting-row">
          <div className="setting-info">
            <span className="setting-label">Use live backend</span>
            <span className="setting-description">
              Connect to real backend at localhost:8000 instead of mock data
            </span>
          </div>
          <input
            type="checkbox"
            className="setting-toggle"
            checked={apiMode === "live"}
            onChange={(e) => setApiMode(e.target.checked ? "live" : "mock")}
          />
        </label>

        <div className="setting-status">
          Current mode: <strong>{apiMode}</strong>
        </div>
      </section>
    </main>
  );
}
```

- [ ] **Step 2: Add route in App.tsx**

Edit `src/App.tsx` to add the import and route:

```typescript
import { Routes, Route, Navigate } from "react-router-dom";
import { NavBar } from "./components/NavBar";
import { HomePage } from "./pages/HomePage";
import { LibraryPage } from "./pages/LibraryPage";
import { SearchPage } from "./pages/SearchPage";
import { SettingsPage } from "./pages/SettingsPage";
import "./App.v2.css";

export default function App() {
  return (
    <div className="app-wrapper">
      <NavBar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/library" element={<LibraryPage />} />
        <Route path="/search/:corpusId" element={<SearchPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
```

- [ ] **Step 3: Link settings button in NavBar**

Edit `src/components/NavBar.tsx`. Add Link import and wrap the settings button:

```typescript
import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { db } from "../lib/storage";

export function NavBar() {
  const location = useLocation();
  const [lastCorpusId, setLastCorpusId] = useState<string | null>(null);

  useEffect(() => {
    db.preferences.get<string>("lastCorpusId").then((id) => {
      if (id) setLastCorpusId(id);
    });
  }, []);

  const isActive = (path: string) => {
    if (path === "/") return location.pathname === "/";
    return location.pathname.startsWith(path);
  };

  const searchPath = lastCorpusId ? `/search/${lastCorpusId}` : null;
  const onSearchPage = location.pathname.startsWith("/search/");

  return (
    <header className="nav-bar">
      <Link to="/" className="nav-brand">
        grep<span>meaning</span>
      </Link>
      <nav className="nav-tabs">
        <Link to="/" className={`nav-tab${isActive("/") && !onSearchPage && location.pathname !== "/library" ? " active" : ""}`}>
          Home
        </Link>
        <Link to="/library" className={`nav-tab${isActive("/library") ? " active" : ""}`}>
          Library
        </Link>
        {searchPath ? (
          <Link to={searchPath} className={`nav-tab${onSearchPage ? " active" : ""}`}>
            Search
          </Link>
        ) : (
          <span className="nav-tab disabled">Search</span>
        )}
      </nav>
      <div className="nav-spacer" />
      <button className="nav-icon-btn" aria-label="Help">
        <svg viewBox="0 0 20 20" fill="currentColor" width="18" height="18">
          <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM8.94 6.94a.75.75 0 11-1.061-1.061 3 3 0 112.871 5.026v.345a.75.75 0 01-1.5 0v-.5c0-.72.57-1.172 1.081-1.287A1.5 1.5 0 108.94 6.94zM10 15a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
        </svg>
      </button>
      <Link to="/settings" className={`nav-icon-btn${isActive("/settings") ? " active" : ""}`} aria-label="Settings">
        <svg viewBox="0 0 20 20" fill="currentColor" width="18" height="18">
          <path fillRule="evenodd" d="M7.84 1.804A1 1 0 018.82 1h2.36a1 1 0 01.98.804l.331 1.652a6.993 6.993 0 011.929 1.115l1.598-.54a1 1 0 011.186.447l1.18 2.044a1 1 0 01-.205 1.251l-1.267 1.113a7.047 7.047 0 010 2.228l1.267 1.113a1 1 0 01.206 1.25l-1.18 2.045a1 1 0 01-1.187.447l-1.598-.54a6.993 6.993 0 01-1.929 1.115l-.33 1.652a1 1 0 01-.98.804H8.82a1 1 0 01-.98-.804l-.331-1.652a6.993 6.993 0 01-1.929-1.115l-1.598.54a1 1 0 01-1.186-.447l-1.18-2.044a1 1 0 01.205-1.251l1.267-1.114a7.05 7.05 0 010-2.227L1.821 7.773a1 1 0 01-.206-1.25l1.18-2.045a1 1 0 011.187-.447l1.598.54A6.993 6.993 0 017.51 3.456l.33-1.652zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
        </svg>
      </Link>
    </header>
  );
}
```

- [ ] **Step 4: Add settings CSS**

Add to `src/App.v2.css` (append to existing file):

```css
/* Settings page */
.settings-section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.5rem;
  max-width: 600px;
}

.setting-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 0;
  border-bottom: 1px solid var(--border);
  cursor: pointer;
}

.setting-row:last-of-type {
  border-bottom: none;
}

.setting-info {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.setting-label {
  font-weight: 500;
  color: var(--text);
}

.setting-description {
  font-size: 0.875rem;
  color: var(--text-muted);
}

.setting-toggle {
  width: 44px;
  height: 24px;
  appearance: none;
  background: var(--border);
  border-radius: 12px;
  position: relative;
  cursor: pointer;
  transition: background 0.2s;
}

.setting-toggle::before {
  content: "";
  position: absolute;
  top: 2px;
  left: 2px;
  width: 20px;
  height: 20px;
  background: white;
  border-radius: 50%;
  transition: transform 0.2s;
}

.setting-toggle:checked {
  background: var(--accent);
}

.setting-toggle:checked::before {
  transform: translateX(20px);
}

.setting-status {
  margin-top: 1rem;
  padding: 0.75rem;
  background: var(--surface-alt);
  border-radius: 4px;
  font-size: 0.875rem;
  color: var(--text-muted);
}

.nav-icon-btn.active {
  color: var(--accent);
}
```

- [ ] **Step 5: Verify the page renders**

Run: `cd frontend && npm run dev`
Navigate to http://localhost:5173/settings
Expected: Settings page with toggle visible

- [ ] **Step 6: Run type check**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 7: Commit**

```bash
git add frontend/src/pages/SettingsPage.tsx frontend/src/App.tsx frontend/src/components/NavBar.tsx frontend/src/App.v2.css
git commit -m "feat: add settings page with API mode toggle"
```

---

### Task 3: Built-in Corpora Section on Homepage

**Files:**
- Modify: `src/pages/HomePage.tsx`

**Interfaces:**
- Consumes: None (static data)
- Produces: "Built-in Corpora" section with demo and browsecomp cards

- [ ] **Step 1: Modify HomePage to add Built-in Corpora section**

Edit `src/pages/HomePage.tsx`. Replace the quick-start section:

```typescript
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
```

- [ ] **Step 2: Add CSS for built-in corpora cards**

Append to `src/App.v2.css`:

```css
/* Built-in corpora section */
.builtin-corpora-section {
  margin-bottom: 2rem;
}

.builtin-corpora-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
}

.builtin-corpus-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.25rem;
  cursor: pointer;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.builtin-corpus-card:hover {
  border-color: var(--accent);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.builtin-corpus-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.5rem;
}

.builtin-corpus-name {
  font-weight: 600;
  font-size: 1.1rem;
  color: var(--text);
}

.builtin-corpus-badge {
  font-size: 0.75rem;
  padding: 0.25rem 0.5rem;
  background: var(--accent-light);
  color: var(--accent);
  border-radius: 4px;
  font-weight: 500;
}

.builtin-corpus-description {
  color: var(--text-muted);
  font-size: 0.9rem;
  margin: 0 0 1rem 0;
  line-height: 1.4;
}

.btn-secondary {
  background: transparent;
  border: 1px solid var(--border);
  padding: 0.5rem 1rem;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.875rem;
  color: var(--text);
  transition: background 0.2s, border-color 0.2s;
}

.btn-secondary:hover {
  background: var(--surface-alt);
  border-color: var(--accent);
}
```

- [ ] **Step 3: Verify the homepage renders correctly**

Run: `cd frontend && npm run dev`
Navigate to http://localhost:5173/
Expected: "Built-in Corpora" section with Demo and BrowseComp cards

- [ ] **Step 4: Run type check**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/HomePage.tsx frontend/src/App.v2.css
git commit -m "feat: add built-in corpora section to homepage"
```

---

### Task 4: Remove Auto-Execute Query

**Files:**
- Modify: `src/hooks/useDashboard.ts`
- Modify: `src/pages/SearchPage.tsx`

**Interfaces:**
- Modifies: `useDashboard` hook - removes `seedQuery` parameter, removes auto-run on mount
- Modifies: `SearchPage` - empty input placeholder, no default query

- [ ] **Step 1: Modify useDashboard to remove seedQuery and auto-run**

Edit `src/hooks/useDashboard.ts`. Remove the seedQuery parameter and auto-run useEffect:

```typescript
import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "../lib/api";
import {
  ScoreCache,
  matchedCount,
  rankedSlice,
  recutFacets,
  recutHistogram,
  type CachedScore,
} from "../lib/scoreCache";
import type { Chip, Facets, FreshDocument, HistogramBin, QueryEvent, RefineEvent } from "../lib/types";

export type LatencyKind = "cold" | "warm" | "cached";
export type Tab = "rel" | "foot" | "perf";

const EMPTY_FACETS: Facets = { type: [], category: [], year: [] };
const FEED_LIMIT = 200;

export interface DashboardView {
  histogram: HistogramBin[];
  facets: Facets;
  matched: number;
  results: CachedScore[];
}

function viewFromCache(cache: ScoreCache, threshold: number): DashboardView {
  const all = cache.all();
  return {
    histogram: recutHistogram(all),
    facets: recutFacets(all, threshold),
    matched: matchedCount(all, threshold),
    results: rankedSlice(all, FEED_LIMIT, threshold),
  };
}

function readFileText(file: File): Promise<string> {
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

const EMPTY_VIEW: DashboardView = {
  histogram: recutHistogram([]),
  facets: EMPTY_FACETS,
  matched: 0,
  results: [],
};

export function useDashboard() {
  const cacheRef = useRef(new ScoreCache());
  const abortRef = useRef<AbortController | null>(null);
  const refineAbortRef = useRef<AbortController | null>(null);
  const chipSnapshotsRef = useRef(new Map<string, CachedScore[]>());
  const thresholdRef = useRef(0.5);

  const [predicate, setPredicate] = useState("");
  const [threshold, setThresholdState] = useState(0.5);
  const [hasRun, setHasRun] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [refining, setRefining] = useState(false);
  const [scanned, setScanned] = useState(0);
  const [etaMs, setEtaMs] = useState(0);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [docsPerSec, setDocsPerSec] = useState(0);
  const [latencyMs, setLatencyMs] = useState(0);
  const [latencyKind, setLatencyKind] = useState<LatencyKind>("cold");
  const [latHistory, setLatHistory] = useState<number[]>([]);
  const [view, setView] = useState<DashboardView>(EMPTY_VIEW);
  const [chips, setChips] = useState<Chip[]>([]);
  const [activeTab, setActiveTab] = useState<Tab>("rel");

  const pushLatency = useCallback((ms: number) => {
    setLatHistory((prev) => [...prev, ms].slice(-16));
  }, []);

  const runQuery = useCallback(
    async (nextPredicate: string) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      cacheRef.current.clear();
      chipSnapshotsRef.current.clear();
      setHasRun(true);
      setStreaming(true);
      setScanned(0);
      setEtaMs(0);
      setView(EMPTY_VIEW);

      const startedAt = performance.now();

      const onEvent = (event: QueryEvent) => {
        if (controller.signal.aborted) return;
        if (event.type === "result") {
          cacheRef.current.upsert(event);
          return;
        }
        if (event.type === "aggregate") {
          setScanned(event.scanned);
          setEtaMs(event.eta_ms);
          setView(viewFromCache(cacheRef.current, thresholdRef.current));
          return;
        }
        if (event.type === "done") {
          const ms = event.elapsed_ms || Math.round(performance.now() - startedAt);
          setElapsedMs(ms);
          setScanned(event.scanned);
          setDocsPerSec(ms ? Math.round((event.scanned / ms) * 1000) : 0);
          setView(viewFromCache(cacheRef.current, thresholdRef.current));
          setLatencyMs(ms);
          setLatencyKind("cold");
          pushLatency(ms);
        }
      };

      try {
        await api.query(
          { predicate: nextPredicate, threshold: thresholdRef.current },
          onEvent,
          controller.signal,
        );
      } finally {
        if (abortRef.current === controller) setStreaming(false);
      }
    },
    [pushLatency],
  );

  const setThreshold = useCallback(
    (next: number) => {
      const clamped = Math.max(0, Math.min(1, next));
      thresholdRef.current = clamped;
      setThresholdState(clamped);
      setView(viewFromCache(cacheRef.current, clamped));
      setLatencyMs(5);
      setLatencyKind("cached");
      pushLatency(5);
    },
    [pushLatency],
  );

  const applyRefineEvent = useCallback(
    (event: RefineEvent, snapshot: CachedScore[]) => {
      if (event.type === "chip") {
        chipSnapshotsRef.current.set(event.chip.clause_id, snapshot);
        setChips((prev) => [...prev, event.chip]);
        setLatencyMs(event.refine_ms);
        setLatencyKind(event.latency_kind);
        return;
      }
      if (event.type === "diff") {
        for (const chunkId of event.removed) cacheRef.current.remove(chunkId);
        for (const item of event.rescored) cacheRef.current.updateScore(item.chunk_id, item.score);
        for (const item of event.added) cacheRef.current.upsert(item);
        setView(viewFromCache(cacheRef.current, thresholdRef.current));
        return;
      }
      if (event.type === "aggregate") {
        setScanned(event.scanned);
        setEtaMs(event.eta_ms);
        setView(viewFromCache(cacheRef.current, thresholdRef.current));
        return;
      }
      if (event.type === "done") {
        setElapsedMs(event.elapsed_ms);
        setDocsPerSec(event.elapsed_ms ? Math.round((event.scanned / event.elapsed_ms) * 1000) : 0);
        setLatencyMs(event.elapsed_ms);
        setLatencyKind(event.warm ? "warm" : "cold");
        pushLatency(event.elapsed_ms);
      }
    },
    [pushLatency],
  );

  const runRefineRequest = useCallback(
    async (request: Parameters<typeof api.refine>[0]) => {
      refineAbortRef.current?.abort();
      const controller = new AbortController();
      refineAbortRef.current = controller;
      const snapshot = cacheRef.current.all();
      setRefining(true);
      try {
        await api.refine(request, (event) => {
          if (controller.signal.aborted) return;
          applyRefineEvent(event, snapshot);
        }, controller.signal);
      } finally {
        if (refineAbortRef.current === controller) setRefining(false);
      }
    },
    [applyRefineEvent],
  );

  const runRefine = useCallback(
    async (utterance: string) => {
      const trimmed = utterance.trim();
      if (!trimmed) return;
      await runRefineRequest({ utterance: trimmed });
    },
    [runRefineRequest],
  );

  const runClickRefine = useCallback(
    async (chunkId: string, sign: "+" | "-") => {
      await runRefineRequest({ click: { chunk_id: chunkId, sign } });
    },
    [runRefineRequest],
  );

  const removeChip = useCallback(
    async (clauseId: string) => {
      const response = await api.deleteClause(clauseId);
      if (!response.removed) return;
      const snapshot = chipSnapshotsRef.current.get(clauseId);
      if (snapshot) {
        cacheRef.current.replaceAll(snapshot);
        setView(viewFromCache(cacheRef.current, thresholdRef.current));
      }
      setChips((prev) => {
        const index = prev.findIndex((chip) => chip.clause_id === clauseId);
        if (index === -1) return prev;
        for (const chip of prev.slice(index)) chipSnapshotsRef.current.delete(chip.clause_id);
        return prev.slice(0, index);
      });
      setLatencyMs(response.refine_ms);
      setLatencyKind("cached");
      pushLatency(response.refine_ms);
    },
    [pushLatency],
  );

  const ingestFreshFiles = useCallback(
    async (files: File[] | FileList) => {
      const documents: FreshDocument[] = await Promise.all(
        Array.from(files).map(async (file) => ({
          title: file.name,
          text: await readFileText(file),
          type: "code",
          category: file.name.split(".").pop() || "fresh",
          year: new Date().getFullYear(),
          path: file.name,
          lang: file.name.split(".").pop() || null,
          repo: "fresh",
        })),
      );
      if (!documents.length) return;
      await api.ingest("demo", documents);
      await runQuery(predicate);
    },
    [predicate, runQuery],
  );

  const ingestCorpus = useCallback(
    async (corpusId: "demo" | "browsecomp", limit?: number) => {
      cacheRef.current.clear();
      chipSnapshotsRef.current.clear();
      setChips([]);
      setView(EMPTY_VIEW);
      setHasRun(false);
      await api.ingest(corpusId, [], limit);
    },
    [],
  );

  return {
    predicate,
    setPredicate,
    threshold,
    setThreshold,
    hasRun,
    streaming,
    refining,
    scanned,
    etaMs,
    elapsedMs,
    docsPerSec,
    latencyMs,
    latencyKind,
    latHistory,
    view,
    chips,
    activeTab,
    setActiveTab,
    runQuery,
    runRefine,
    runClickRefine,
    removeChip,
    ingestFreshFiles,
    ingestCorpus,
    mode: api.mode,
  };
}
```

- [ ] **Step 2: Update SearchPage to use empty predicate and placeholder**

Edit `src/pages/SearchPage.tsx`. Remove DEFAULT_QUERY, update useDashboard call, and update placeholder:

At line 16, remove the DEFAULT_QUERY constant.

At line 25, change:
```typescript
const d = useDashboard(DEFAULT_QUERY);
```
to:
```typescript
const d = useDashboard();
```

At line 157, the placeholder is already "Describe what you're looking for..." which is good, but we can make it simpler:
```typescript
placeholder="Type a query..."
```

- [ ] **Step 3: Run type check**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 4: Run tests**

Run: `cd frontend && npm test -- --run`
Expected: All tests pass

- [ ] **Step 5: Verify behavior**

Run: `cd frontend && npm run dev`
Navigate to http://localhost:5173/search/demo
Expected: Empty query input, empty results area with "No scan yet" message

- [ ] **Step 6: Commit**

```bash
git add frontend/src/hooks/useDashboard.ts frontend/src/pages/SearchPage.tsx
git commit -m "feat: remove auto-execute query on page load"
```

---

### Task 5: Document Preview Component

**Files:**
- Create: `src/components/DocumentPreview.tsx`
- Modify: `src/pages/SearchPage.tsx`

**Interfaces:**
- Produces: `DocumentPreview` component with props `{ result: CachedScore; onClose: () => void }`
- Produces: `DocumentFullPreview` component with props `{ result: CachedScore; onClose: () => void }`

- [ ] **Step 1: Create DocumentPreview component**

Create `src/components/DocumentPreview.tsx`:

```typescript
import { useState } from "react";
import type { CachedScore } from "../lib/scoreCache";

interface DocumentPreviewProps {
  result: CachedScore;
  onClose: () => void;
}

export function DocumentPreview({ result, onClose }: DocumentPreviewProps) {
  const [showFull, setShowFull] = useState(false);

  if (showFull) {
    return <DocumentFullPreview result={result} onClose={() => setShowFull(false)} onBack={() => setShowFull(false)} />;
  }

  return (
    <div className="preview-overlay" onClick={onClose}>
      <div className="preview-card" onClick={(e) => e.stopPropagation()}>
        <button className="preview-close" onClick={onClose} aria-label="Close">
          &times;
        </button>

        <div className="preview-header">
          <span className={`type-badge type-${result.meta.type ?? "code"}`}>
            {result.meta.type ?? "code"}
          </span>
          <h2 className="preview-title">{result.meta.title}</h2>
        </div>

        <div className="preview-meta-grid">
          <div className="preview-meta-item">
            <span className="preview-meta-label">Category</span>
            <span className="preview-meta-value">{result.meta.category ?? "—"}</span>
          </div>
          <div className="preview-meta-item">
            <span className="preview-meta-label">Year</span>
            <span className="preview-meta-value">{result.meta.year ?? "—"}</span>
          </div>
          {result.meta.path && (
            <div className="preview-meta-item">
              <span className="preview-meta-label">Path</span>
              <span className="preview-meta-value">{result.meta.path}</span>
            </div>
          )}
          {result.meta.repo && (
            <div className="preview-meta-item">
              <span className="preview-meta-label">Repo</span>
              <span className="preview-meta-value">{result.meta.repo}</span>
            </div>
          )}
          {result.meta.lang && (
            <div className="preview-meta-item">
              <span className="preview-meta-label">Language</span>
              <span className="preview-meta-value">{result.meta.lang}</span>
            </div>
          )}
        </div>

        <div className="preview-score">
          <span className="preview-score-label">Relevance Score</span>
          <div className="preview-score-bar">
            <div className="preview-score-fill" style={{ width: `${Math.round(result.score * 100)}%` }} />
          </div>
          <span className="preview-score-value">{result.score.toFixed(3)}</span>
        </div>

        <div className="preview-actions">
          <button className="btn-primary" onClick={() => setShowFull(true)}>
            Preview Document
          </button>
        </div>
      </div>
    </div>
  );
}

interface DocumentFullPreviewProps {
  result: CachedScore;
  onClose: () => void;
  onBack: () => void;
}

function DocumentFullPreview({ result, onClose, onBack }: DocumentFullPreviewProps) {
  return (
    <div className="preview-overlay" onClick={onClose}>
      <div className="preview-full-panel" onClick={(e) => e.stopPropagation()}>
        <div className="preview-full-header">
          <button className="preview-back" onClick={onBack} aria-label="Back">
            &larr; Back
          </button>
          <h2 className="preview-title">{result.meta.title}</h2>
          <button className="preview-close" onClick={onClose} aria-label="Close">
            &times;
          </button>
        </div>

        <div className="preview-full-meta">
          <span className={`type-badge type-${result.meta.type ?? "code"}`}>
            {result.meta.type ?? "code"}
          </span>
          <span>{result.meta.category ?? "—"}</span>
          <span>{result.meta.year ?? "—"}</span>
          {result.meta.path && <span>{result.meta.path}</span>}
        </div>

        <div className="preview-full-content">
          <p className="preview-placeholder">
            Document content preview not available.
            <br />
            <br />
            The full document text is stored in the backend and would require an additional API call to fetch.
          </p>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Add CSS for document preview**

Append to `src/App.v2.css`:

```css
/* Document preview */
.preview-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 1rem;
}

.preview-card {
  background: var(--surface);
  border-radius: 12px;
  padding: 1.5rem;
  max-width: 480px;
  width: 100%;
  position: relative;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}

.preview-close {
  position: absolute;
  top: 1rem;
  right: 1rem;
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: var(--text-muted);
  padding: 0.25rem;
  line-height: 1;
}

.preview-close:hover {
  color: var(--text);
}

.preview-header {
  margin-bottom: 1.5rem;
}

.preview-title {
  font-size: 1.25rem;
  font-weight: 600;
  margin: 0.5rem 0 0 0;
  color: var(--text);
}

.preview-meta-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.preview-meta-item {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.preview-meta-label {
  font-size: 0.75rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.preview-meta-value {
  font-size: 0.9rem;
  color: var(--text);
  word-break: break-word;
}

.preview-score {
  background: var(--surface-alt);
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 1.5rem;
}

.preview-score-label {
  font-size: 0.75rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  display: block;
  margin-bottom: 0.5rem;
}

.preview-score-bar {
  height: 8px;
  background: var(--border);
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 0.5rem;
}

.preview-score-fill {
  height: 100%;
  background: var(--accent);
  border-radius: 4px;
}

.preview-score-value {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text);
}

.preview-actions {
  display: flex;
  gap: 0.75rem;
}

/* Full preview panel */
.preview-full-panel {
  background: var(--surface);
  border-radius: 12px;
  max-width: 800px;
  width: 100%;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}

.preview-full-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem 1.5rem;
  border-bottom: 1px solid var(--border);
}

.preview-back {
  background: none;
  border: none;
  color: var(--accent);
  cursor: pointer;
  font-size: 0.9rem;
  padding: 0.25rem 0.5rem;
}

.preview-back:hover {
  text-decoration: underline;
}

.preview-full-header .preview-title {
  flex: 1;
  margin: 0;
  font-size: 1.1rem;
}

.preview-full-header .preview-close {
  position: static;
}

.preview-full-meta {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.75rem 1.5rem;
  background: var(--surface-alt);
  font-size: 0.875rem;
  color: var(--text-muted);
}

.preview-full-content {
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem;
}

.preview-placeholder {
  color: var(--text-muted);
  text-align: center;
  padding: 3rem 1rem;
}

/* Make result rows clickable */
.result-row {
  cursor: pointer;
}

.result-row:hover {
  background: var(--surface-alt);
}
```

- [ ] **Step 3: Integrate preview into SearchPage**

Edit `src/pages/SearchPage.tsx`. Add state for selected document and preview component:

Add import at top:
```typescript
import { DocumentPreview } from "../components/DocumentPreview";
```

Add state in SearchPage component (after line 24):
```typescript
const [previewResult, setPreviewResult] = useState<CachedScore | null>(null);
```

Add import for CachedScore at top:
```typescript
import type { CachedScore } from "../lib/scoreCache";
```

Pass handler to TabbedSection (around line 117):
```typescript
onResultClick={(result) => setPreviewResult(result)}
```

Add preview modal before closing `</main>` tag:
```typescript
{previewResult && (
  <DocumentPreview result={previewResult} onClose={() => setPreviewResult(null)} />
)}
```

Update TabbedSectionProps interface to include onResultClick:
```typescript
interface TabbedSectionProps {
  // ... existing props
  onResultClick: (result: CachedScore) => void;
}
```

Update TabbedSection function parameters and pass to ResultList:
```typescript
function TabbedSection({
  // ... existing params
  onResultClick,
}: TabbedSectionProps) {
  // ... existing code
  {tab === "results" && (
    <ResultList results={results} threshold={threshold} hasRun={hasRun} onClickRefine={onClickRefine} onResultClick={onResultClick} />
  )}
```

Update ResultList props and add click handler:
```typescript
function ResultList({
  results,
  threshold,
  hasRun,
  onClickRefine,
  onResultClick,
}: {
  results: CachedScore[];
  threshold: number;
  hasRun: boolean;
  onClickRefine: (chunkId: string, sign: "+" | "-") => Promise<void>;
  onResultClick: (result: CachedScore) => void;
}) {
```

Update the article element to handle clicks (but not on action buttons):
```typescript
<article
  className={`result-row${matched ? " matched" : ""}`}
  key={result.chunk_id}
  onClick={() => onResultClick(result)}
>
```

Update action buttons to stop propagation:
```typescript
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
```

- [ ] **Step 4: Run type check**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 5: Verify preview works**

Run: `cd frontend && npm run dev`
Navigate to http://localhost:5173/search/demo
Enter a query, click Scan
Click on a result row
Expected: Metadata card appears. Click "Preview Document" to see full panel.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/DocumentPreview.tsx frontend/src/pages/SearchPage.tsx frontend/src/App.v2.css
git commit -m "feat: add document preview on result click"
```

---

### Task 6: Final Integration Test

**Files:**
- None (manual verification)

- [ ] **Step 1: Start backend with Modal**

```bash
SCORER_BACKEND=modal /opt/miniconda3/bin/python3 -m uvicorn backend.main:app --port 8000
```

- [ ] **Step 2: Start frontend**

```bash
cd frontend && npm run dev
```

- [ ] **Step 3: Verify settings toggle**

1. Navigate to http://localhost:5173/settings
2. Toggle "Use live backend" on
3. Verify current mode shows "live"

- [ ] **Step 4: Verify built-in corpora**

1. Navigate to http://localhost:5173/
2. Verify "Built-in Corpora" section shows Demo and BrowseComp cards
3. Click Demo card → navigates to /search/demo

- [ ] **Step 5: Verify no auto-query**

1. On /search/demo page, verify query input is empty
2. Verify results area shows "No scan yet"
3. Enter a query and click Scan → results appear

- [ ] **Step 6: Verify document preview**

1. Click on a result row → metadata card appears
2. Click "Preview Document" → full panel appears
3. Click outside or X → closes

- [ ] **Step 7: Run all tests**

```bash
cd frontend && npm test -- --run
```
Expected: All tests pass

- [ ] **Step 8: Final commit**

```bash
git add -A
git commit -m "test: verify frontend UX improvements integration"
```
