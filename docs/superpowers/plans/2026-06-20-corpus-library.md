# Corpus Library & Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the single-page search UI into a multi-page application with corpus management, dashboard, and persistent IndexedDB storage.

**Architecture:** React Router for navigation between Home/Library/Search pages. IndexedDB (via `idb` library) stores corpus metadata, saved queries, hints, and preferences. Existing search functionality in App.v2.tsx becomes the SearchPage component with corpus-awareness.

**Tech Stack:** React 19, React Router 6, idb (IndexedDB wrapper), TypeScript, Vite

## Global Constraints

- Apple HIG styling: system fonts, colors, 8pt grid, 44pt touch targets
- All new styles added to existing `App.v2.css`
- Corpus IDs must be URL-safe slugs
- Demo corpus is read-only (no edit/delete)
- No backend changes required

---

## File Structure

```
frontend/src/
├── main.tsx                    # MODIFY: Add BrowserRouter wrapper, DB init
├── App.tsx                     # CREATE: Router outlet with NavBar
├── pages/
│   ├── HomePage.tsx            # CREATE: Dashboard with favorites, recent, stats
│   ├── LibraryPage.tsx         # CREATE: Corpus list with CRUD
│   └── SearchPage.tsx          # CREATE: Adapted from App.v2.tsx
├── components/
│   ├── NavBar.tsx              # CREATE: Top navigation tabs
│   ├── CorpusCard.tsx          # CREATE: Reusable corpus display
│   ├── CreateCorpusModal.tsx   # CREATE: New corpus form with file drop
│   ├── EditCorpusModal.tsx     # CREATE: Edit corpus metadata
│   ├── SavedQueriesDrawer.tsx  # CREATE: Collapsible saved queries
│   └── Hint.tsx                # CREATE: Contextual hint component
├── hooks/
│   ├── useDashboard.ts         # MODIFY: Accept corpusId param
│   ├── useCorpora.ts           # CREATE: Corpus CRUD operations
│   ├── useSavedQueries.ts      # CREATE: Saved query operations
│   └── useHints.ts             # CREATE: Hint visibility state
├── lib/
│   ├── storage.ts              # CREATE: IndexedDB wrapper
│   ├── types.ts                # MODIFY: Add Corpus, SavedQuery types
│   └── slugify.ts              # CREATE: URL-safe slug generator
└── App.v2.css                  # MODIFY: Add new page/component styles
```

---

### Task 1: Install Dependencies and Add Types

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/src/lib/types.ts`
- Create: `frontend/src/lib/slugify.ts`

**Interfaces:**
- Produces: `Corpus`, `SavedQuery` types used by all subsequent tasks
- Produces: `slugify(name: string): string` function

- [ ] **Step 1: Install react-router-dom and idb**

```bash
cd frontend && npm install react-router-dom idb
```

Expected: packages added to package.json dependencies

- [ ] **Step 2: Add Corpus and SavedQuery types to types.ts**

Add to the end of `frontend/src/lib/types.ts`:

```typescript
// Corpus management types
export interface Corpus {
  id: string;
  name: string;
  description: string;
  tags: string[];
  createdAt: number;
  lastUsedAt: number;
  isFavorite: boolean;
  isDemo: boolean;
  documentCount: number;
  source: "files" | "demo";
}

export interface SavedQuery {
  id: string;
  corpusId: string;
  predicate: string;
  threshold: number;
  chips: Chip[];
  name: string;
  notes: string;
  savedAt: number;
}
```

- [ ] **Step 3: Create slugify utility**

Create `frontend/src/lib/slugify.ts`:

```typescript
export function slugify(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 50);
}

export function generateId(): string {
  return crypto.randomUUID();
}
```

- [ ] **Step 4: Verify types compile**

```bash
cd frontend && npm run build
```

Expected: Build succeeds with no type errors

- [ ] **Step 5: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/src/lib/types.ts frontend/src/lib/slugify.ts
git commit -m "feat: add corpus types and install router/idb dependencies"
```

---

### Task 2: IndexedDB Storage Layer

**Files:**
- Create: `frontend/src/lib/storage.ts`
- Create: `frontend/src/lib/storage.test.ts`

**Interfaces:**
- Produces: `initDB(): Promise<void>` - initializes database and seeds demo corpus
- Produces: `db.corpora.getAll()`, `db.corpora.get(id)`, `db.corpora.put(corpus)`, `db.corpora.delete(id)`, `db.corpora.getFavorites()`, `db.corpora.getRecent(limit)`
- Produces: `db.savedQueries.getByCorpus(corpusId)`, `db.savedQueries.put(query)`, `db.savedQueries.delete(id)`
- Produces: `db.hints.isDismissed(key)`, `db.hints.dismiss(key)`
- Produces: `db.preferences.get(key)`, `db.preferences.set(key, value)`

- [ ] **Step 1: Write failing test for storage initialization**

Create `frontend/src/lib/storage.test.ts`:

```typescript
import { describe, it, expect, beforeEach } from "vitest";
import { initDB, db, clearDB } from "./storage";

describe("storage", () => {
  beforeEach(async () => {
    await clearDB();
  });

  it("initializes with demo corpus", async () => {
    await initDB();
    const demo = await db.corpora.get("demo");
    expect(demo).toBeDefined();
    expect(demo?.name).toBe("Demo Corpus");
    expect(demo?.isDemo).toBe(true);
  });

  it("persists and retrieves a corpus", async () => {
    await initDB();
    const corpus = {
      id: "test-corpus",
      name: "Test Corpus",
      description: "A test",
      tags: ["test"],
      createdAt: Date.now(),
      lastUsedAt: Date.now(),
      isFavorite: false,
      isDemo: false,
      documentCount: 10,
      source: "files" as const,
    };
    await db.corpora.put(corpus);
    const retrieved = await db.corpora.get("test-corpus");
    expect(retrieved).toEqual(corpus);
  });

  it("returns recent corpora sorted by lastUsedAt", async () => {
    await initDB();
    const now = Date.now();
    await db.corpora.put({
      id: "old", name: "Old", description: "", tags: [], createdAt: now - 2000,
      lastUsedAt: now - 2000, isFavorite: false, isDemo: false, documentCount: 1, source: "files",
    });
    await db.corpora.put({
      id: "new", name: "New", description: "", tags: [], createdAt: now - 1000,
      lastUsedAt: now - 1000, isFavorite: false, isDemo: false, documentCount: 1, source: "files",
    });
    const recent = await db.corpora.getRecent(2);
    expect(recent[0].id).toBe("new");
    expect(recent[1].id).toBe("old");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd frontend && npm test -- storage.test.ts
```

Expected: FAIL - module not found

- [ ] **Step 3: Implement storage layer**

Create `frontend/src/lib/storage.ts`:

```typescript
import { openDB, deleteDB, type IDBPDatabase } from "idb";
import type { Corpus, SavedQuery } from "./types";

const DB_NAME = "grepmeaning-db";
const DB_VERSION = 1;

interface HintRecord {
  key: string;
  dismissed: boolean;
}

interface PreferenceRecord {
  key: string;
  value: unknown;
}

let dbInstance: IDBPDatabase | null = null;

async function getDB(): Promise<IDBPDatabase> {
  if (dbInstance) return dbInstance;
  dbInstance = await openDB(DB_NAME, DB_VERSION, {
    upgrade(database) {
      // Corpora store
      const corporaStore = database.createObjectStore("corpora", { keyPath: "id" });
      corporaStore.createIndex("by-lastUsedAt", "lastUsedAt");
      corporaStore.createIndex("by-isFavorite", "isFavorite");
      corporaStore.createIndex("by-createdAt", "createdAt");

      // Saved queries store
      const queriesStore = database.createObjectStore("savedQueries", { keyPath: "id" });
      queriesStore.createIndex("by-corpusId", "corpusId");
      queriesStore.createIndex("by-savedAt", "savedAt");

      // Hints store
      database.createObjectStore("hints", { keyPath: "key" });

      // Preferences store
      database.createObjectStore("preferences", { keyPath: "key" });
    },
  });
  return dbInstance;
}

const DEMO_CORPUS: Corpus = {
  id: "demo",
  name: "Demo Corpus",
  description: "Sample papers and code to explore the tool",
  tags: [],
  createdAt: Date.now(),
  lastUsedAt: Date.now(),
  isFavorite: false,
  isDemo: true,
  documentCount: 24,
  source: "demo",
};

export async function initDB(): Promise<void> {
  const database = await getDB();
  const existing = await database.get("corpora", "demo");
  if (!existing) {
    await database.put("corpora", DEMO_CORPUS);
  }
}

export async function clearDB(): Promise<void> {
  dbInstance = null;
  await deleteDB(DB_NAME);
}

export const db = {
  corpora: {
    async getAll(): Promise<Corpus[]> {
      const database = await getDB();
      return database.getAll("corpora");
    },
    async get(id: string): Promise<Corpus | undefined> {
      const database = await getDB();
      return database.get("corpora", id);
    },
    async put(corpus: Corpus): Promise<void> {
      const database = await getDB();
      await database.put("corpora", corpus);
    },
    async delete(id: string): Promise<void> {
      const database = await getDB();
      await database.delete("corpora", id);
    },
    async getFavorites(): Promise<Corpus[]> {
      const database = await getDB();
      const all = await database.getAll("corpora");
      return all.filter((c) => c.isFavorite).sort((a, b) => b.lastUsedAt - a.lastUsedAt);
    },
    async getRecent(limit: number): Promise<Corpus[]> {
      const database = await getDB();
      const all = await database.getAll("corpora");
      return all.sort((a, b) => b.lastUsedAt - a.lastUsedAt).slice(0, limit);
    },
  },
  savedQueries: {
    async getByCorpus(corpusId: string): Promise<SavedQuery[]> {
      const database = await getDB();
      const index = database.transaction("savedQueries").store.index("by-corpusId");
      return index.getAll(corpusId);
    },
    async put(query: SavedQuery): Promise<void> {
      const database = await getDB();
      await database.put("savedQueries", query);
    },
    async delete(id: string): Promise<void> {
      const database = await getDB();
      await database.delete("savedQueries", id);
    },
    async countByCorpus(corpusId: string): Promise<number> {
      const queries = await this.getByCorpus(corpusId);
      return queries.length;
    },
    async countAll(): Promise<number> {
      const database = await getDB();
      return database.count("savedQueries");
    },
  },
  hints: {
    async isDismissed(key: string): Promise<boolean> {
      const database = await getDB();
      const record = await database.get("hints", key) as HintRecord | undefined;
      return record?.dismissed ?? false;
    },
    async dismiss(key: string): Promise<void> {
      const database = await getDB();
      await database.put("hints", { key, dismissed: true });
    },
  },
  preferences: {
    async get<T>(key: string): Promise<T | undefined> {
      const database = await getDB();
      const record = await database.get("preferences", key) as PreferenceRecord | undefined;
      return record?.value as T | undefined;
    },
    async set<T>(key: string, value: T): Promise<void> {
      const database = await getDB();
      await database.put("preferences", { key, value });
    },
  },
};
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd frontend && npm test -- storage.test.ts
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/storage.ts frontend/src/lib/storage.test.ts
git commit -m "feat: add IndexedDB storage layer with corpus and query persistence"
```

---

### Task 3: NavBar Component

**Files:**
- Create: `frontend/src/components/NavBar.tsx`

**Interfaces:**
- Consumes: React Router's `useLocation`, `useNavigate`, `Link`
- Consumes: `db.preferences.get<string>("lastCorpusId")`
- Produces: `<NavBar />` component with Home/Library/Search tabs

- [ ] **Step 1: Create NavBar component**

Create `frontend/src/components/NavBar.tsx`:

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
      <button className="nav-icon-btn" aria-label="Settings">
        <svg viewBox="0 0 20 20" fill="currentColor" width="18" height="18">
          <path fillRule="evenodd" d="M7.84 1.804A1 1 0 018.82 1h2.36a1 1 0 01.98.804l.331 1.652a6.993 6.993 0 011.929 1.115l1.598-.54a1 1 0 011.186.447l1.18 2.044a1 1 0 01-.205 1.251l-1.267 1.113a7.047 7.047 0 010 2.228l1.267 1.113a1 1 0 01.206 1.25l-1.18 2.045a1 1 0 01-1.187.447l-1.598-.54a6.993 6.993 0 01-1.929 1.115l-.33 1.652a1 1 0 01-.98.804H8.82a1 1 0 01-.98-.804l-.331-1.652a6.993 6.993 0 01-1.929-1.115l-1.598.54a1 1 0 01-1.186-.447l-1.18-2.044a1 1 0 01.205-1.251l1.267-1.114a7.05 7.05 0 010-2.227L1.821 7.773a1 1 0 01-.206-1.25l1.18-2.045a1 1 0 011.187-.447l1.598.54A6.993 6.993 0 017.51 3.456l.33-1.652zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
        </svg>
      </button>
    </header>
  );
}
```

- [ ] **Step 2: Add NavBar styles to App.v2.css**

Add to `frontend/src/App.v2.css` after the existing header styles:

```css
/* ============================================
   NavBar (replaces header-bar for multi-page)
   ============================================ */

.nav-bar {
  display: flex;
  align-items: center;
  gap: var(--s2);
  padding: var(--s2) var(--s4);
  background: var(--bg-primary);
  border-bottom: 1px solid var(--separator-opaque);
  height: 52px;
  position: sticky;
  top: 0;
  z-index: 100;
}

.nav-brand {
  font-family: var(--font-mono);
  font-weight: 600;
  font-size: 16px;
  letter-spacing: -0.02em;
  color: var(--label);
  text-decoration: none;
  white-space: nowrap;
}

.nav-brand span {
  color: var(--system-blue);
}

.nav-tabs {
  display: flex;
  gap: var(--s1);
  margin-left: var(--s4);
}

.nav-tab {
  padding: var(--s2) var(--s3);
  border: none;
  background: none;
  color: var(--label-secondary);
  font-size: 14px;
  font-weight: 500;
  text-decoration: none;
  cursor: pointer;
  border-radius: var(--r1);
  transition: background 0.15s ease, color 0.15s ease;
}

.nav-tab:hover {
  background: var(--fill-tertiary);
}

.nav-tab.active {
  background: var(--system-blue);
  color: white;
}

.nav-tab.disabled {
  color: var(--label-quaternary);
  cursor: not-allowed;
}

.nav-spacer {
  flex: 1;
}

.nav-icon-btn {
  width: 36px;
  height: 36px;
  border: none;
  background: none;
  color: var(--label-secondary);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--r1);
  transition: background 0.15s ease;
}

.nav-icon-btn:hover {
  background: var(--fill);
}

.nav-icon-btn:focus-visible {
  outline: 2px solid var(--system-blue);
  outline-offset: 2px;
}
```

- [ ] **Step 3: Verify build**

```bash
cd frontend && npm run build
```

Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/NavBar.tsx frontend/src/App.v2.css
git commit -m "feat: add NavBar component with Home/Library/Search tabs"
```

---

### Task 4: Router Setup and App Shell

**Files:**
- Modify: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`

**Interfaces:**
- Consumes: `initDB()` from storage.ts
- Consumes: `<NavBar />` component
- Produces: Router setup with `/`, `/library`, `/search/:corpusId` routes

- [ ] **Step 1: Update main.tsx with router and DB initialization**

Replace `frontend/src/main.tsx`:

```typescript
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { initDB } from "./lib/storage";

initDB().then(() => {
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </StrictMode>,
  );
});
```

- [ ] **Step 2: Create App.tsx with router outlet**

Create `frontend/src/App.tsx`:

```typescript
import { Routes, Route, Navigate } from "react-router-dom";
import { NavBar } from "./components/NavBar";
import { HomePage } from "./pages/HomePage";
import { LibraryPage } from "./pages/LibraryPage";
import { SearchPage } from "./pages/SearchPage";
import "./App.v2.css";

export default function App() {
  return (
    <div className="app-wrapper">
      <NavBar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/library" element={<LibraryPage />} />
        <Route path="/search/:corpusId" element={<SearchPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
```

- [ ] **Step 3: Create placeholder pages**

Create `frontend/src/pages/HomePage.tsx`:

```typescript
export function HomePage() {
  return (
    <main className="page-content">
      <h1>Home Page</h1>
      <p>Dashboard coming soon...</p>
    </main>
  );
}
```

Create `frontend/src/pages/LibraryPage.tsx`:

```typescript
export function LibraryPage() {
  return (
    <main className="page-content">
      <h1>Library</h1>
      <p>Corpus management coming soon...</p>
    </main>
  );
}
```

Create `frontend/src/pages/SearchPage.tsx`:

```typescript
import { useParams } from "react-router-dom";

export function SearchPage() {
  const { corpusId } = useParams<{ corpusId: string }>();
  return (
    <main className="page-content">
      <h1>Search: {corpusId}</h1>
      <p>Search interface coming soon...</p>
    </main>
  );
}
```

- [ ] **Step 4: Add page-content styles**

Add to `frontend/src/App.v2.css`:

```css
/* ============================================
   Page Layout
   ============================================ */

.page-content {
  flex: 1;
  max-width: 1200px;
  width: 100%;
  margin: 0 auto;
  padding: var(--s4);
}

.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: var(--s4);
}

.page-title {
  font-size: 28px;
  font-weight: 700;
  color: var(--label);
  letter-spacing: -0.02em;
}
```

- [ ] **Step 5: Verify app runs**

```bash
cd frontend && npm run dev
```

Open browser to http://localhost:5173 - should see NavBar and "Home Page" text. Click tabs to navigate.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/main.tsx frontend/src/App.tsx frontend/src/pages/ frontend/src/App.v2.css
git commit -m "feat: add router setup with Home/Library/Search page shells"
```

---

### Task 5: CorpusCard Component

**Files:**
- Create: `frontend/src/components/CorpusCard.tsx`

**Interfaces:**
- Consumes: `Corpus` type
- Produces: `<CorpusCard corpus={...} onClick={...} onStar={...} variant="compact" | "full" />`

- [ ] **Step 1: Create CorpusCard component**

Create `frontend/src/components/CorpusCard.tsx`:

```typescript
import type { Corpus } from "../lib/types";

interface CorpusCardProps {
  corpus: Corpus;
  variant?: "compact" | "full";
  onClick?: () => void;
  onStar?: () => void;
  onEdit?: () => void;
  onDelete?: () => void;
}

function formatRelativeTime(timestamp: number): string {
  const seconds = Math.floor((Date.now() - timestamp) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(timestamp).toLocaleDateString();
}

export function CorpusCard({ corpus, variant = "full", onClick, onStar, onEdit, onDelete }: CorpusCardProps) {
  const isCompact = variant === "compact";

  return (
    <article className={`corpus-card ${variant}`} onClick={onClick}>
      <div className="corpus-card-header">
        <button
          className={`star-btn${corpus.isFavorite ? " active" : ""}`}
          onClick={(e) => { e.stopPropagation(); onStar?.(); }}
          aria-label={corpus.isFavorite ? "Remove from favorites" : "Add to favorites"}
          disabled={corpus.isDemo}
        >
          {corpus.isFavorite ? "★" : "☆"}
        </button>
        <h3 className="corpus-card-title">
          {corpus.name}
          {corpus.isDemo && <span className="demo-badge">built-in</span>}
        </h3>
        <span className="corpus-card-count">{corpus.documentCount} docs</span>
      </div>

      {!isCompact && corpus.description && (
        <p className="corpus-card-desc">{corpus.description}</p>
      )}

      <div className="corpus-card-footer">
        {corpus.tags.length > 0 && (
          <div className="corpus-card-tags">
            {corpus.tags.slice(0, 3).map((tag) => (
              <span key={tag} className="corpus-tag">#{tag}</span>
            ))}
          </div>
        )}
        <span className="corpus-card-time">Last used: {formatRelativeTime(corpus.lastUsedAt)}</span>
      </div>

      {!isCompact && !corpus.isDemo && (
        <div className="corpus-card-actions" onClick={(e) => e.stopPropagation()}>
          <button className="btn-secondary" onClick={onEdit}>Edit</button>
          <button className="btn-danger" onClick={onDelete}>Delete</button>
        </div>
      )}
    </article>
  );
}
```

- [ ] **Step 2: Add CorpusCard styles**

Add to `frontend/src/App.v2.css`:

```css
/* ============================================
   Corpus Card
   ============================================ */

.corpus-card {
  background: var(--bg-primary);
  border-radius: var(--r2);
  padding: var(--s3);
  cursor: pointer;
  transition: box-shadow 0.15s ease, transform 0.1s ease;
  border: 1px solid var(--separator-opaque);
}

.corpus-card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  transform: translateY(-1px);
}

.corpus-card.compact {
  min-width: 160px;
  max-width: 200px;
}

.corpus-card-header {
  display: flex;
  align-items: center;
  gap: var(--s2);
  margin-bottom: var(--s2);
}

.star-btn {
  background: none;
  border: none;
  font-size: 18px;
  color: var(--label-tertiary);
  cursor: pointer;
  padding: 0;
  line-height: 1;
  transition: color 0.15s ease, transform 0.1s ease;
}

.star-btn:hover {
  transform: scale(1.1);
}

.star-btn.active {
  color: var(--system-orange);
}

.star-btn:disabled {
  opacity: 0.3;
  cursor: default;
}

.corpus-card-title {
  flex: 1;
  font-size: 15px;
  font-weight: 600;
  color: var(--label);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.demo-badge {
  font-size: 10px;
  font-weight: 500;
  color: var(--label-tertiary);
  background: var(--fill);
  padding: 2px 6px;
  border-radius: var(--r1);
  margin-left: var(--s2);
  text-transform: uppercase;
  letter-spacing: 0.02em;
}

.corpus-card-count {
  font-size: 12px;
  color: var(--label-tertiary);
  white-space: nowrap;
}

.corpus-card-desc {
  font-size: 13px;
  color: var(--label-secondary);
  line-height: 1.4;
  margin-bottom: var(--s2);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.corpus-card-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--s2);
}

.corpus-card-tags {
  display: flex;
  gap: var(--s1);
  flex-wrap: wrap;
}

.corpus-tag {
  font-size: 11px;
  color: var(--system-blue);
  background: rgba(0, 122, 255, 0.1);
  padding: 2px 6px;
  border-radius: var(--r1);
}

.corpus-card-time {
  font-size: 11px;
  color: var(--label-tertiary);
  white-space: nowrap;
}

.corpus-card-actions {
  display: flex;
  gap: var(--s2);
  margin-top: var(--s3);
  padding-top: var(--s3);
  border-top: 1px solid var(--separator);
}

.btn-secondary {
  flex: 1;
  padding: var(--s2) var(--s3);
  border: 1px solid var(--separator-opaque);
  background: var(--bg-primary);
  color: var(--label);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  border-radius: var(--r1);
  transition: background 0.15s ease;
}

.btn-secondary:hover {
  background: var(--fill-tertiary);
}

.btn-danger {
  flex: 1;
  padding: var(--s2) var(--s3);
  border: none;
  background: rgba(255, 59, 48, 0.1);
  color: var(--system-red);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  border-radius: var(--r1);
  transition: background 0.15s ease;
}

.btn-danger:hover {
  background: rgba(255, 59, 48, 0.2);
}
```

- [ ] **Step 3: Verify build**

```bash
cd frontend && npm run build
```

Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/CorpusCard.tsx frontend/src/App.v2.css
git commit -m "feat: add CorpusCard component with compact and full variants"
```

---

### Task 6: useCorpora Hook

**Files:**
- Create: `frontend/src/hooks/useCorpora.ts`

**Interfaces:**
- Consumes: `db.corpora.*` from storage.ts
- Produces: `useCorpora()` hook returning `{ corpora, favorites, recent, loading, refresh, toggleFavorite, deleteCorpus }`

- [ ] **Step 1: Create useCorpora hook**

Create `frontend/src/hooks/useCorpora.ts`:

```typescript
import { useCallback, useEffect, useState } from "react";
import { db } from "../lib/storage";
import type { Corpus } from "../lib/types";

interface UseCorporaResult {
  corpora: Corpus[];
  favorites: Corpus[];
  recent: Corpus[];
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  toggleFavorite: (id: string) => Promise<void>;
  deleteCorpus: (id: string) => Promise<void>;
  updateCorpus: (corpus: Corpus) => Promise<void>;
  getCorpus: (id: string) => Promise<Corpus | undefined>;
}

export function useCorpora(): UseCorporaResult {
  const [corpora, setCorpora] = useState<Corpus[]>([]);
  const [favorites, setFavorites] = useState<Corpus[]>([]);
  const [recent, setRecent] = useState<Corpus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const [allCorpora, favs, recentList] = await Promise.all([
        db.corpora.getAll(),
        db.corpora.getFavorites(),
        db.corpora.getRecent(5),
      ]);
      setCorpora(allCorpora);
      setFavorites(favs);
      setRecent(recentList);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load corpora");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const toggleFavorite = useCallback(async (id: string) => {
    const corpus = await db.corpora.get(id);
    if (!corpus || corpus.isDemo) return;
    await db.corpora.put({ ...corpus, isFavorite: !corpus.isFavorite });
    await refresh();
  }, [refresh]);

  const deleteCorpus = useCallback(async (id: string) => {
    const corpus = await db.corpora.get(id);
    if (!corpus || corpus.isDemo) return;
    await db.corpora.delete(id);
    await refresh();
  }, [refresh]);

  const updateCorpus = useCallback(async (corpus: Corpus) => {
    if (corpus.isDemo) return;
    await db.corpora.put(corpus);
    await refresh();
  }, [refresh]);

  const getCorpus = useCallback(async (id: string) => {
    return db.corpora.get(id);
  }, []);

  return {
    corpora,
    favorites,
    recent,
    loading,
    error,
    refresh,
    toggleFavorite,
    deleteCorpus,
    updateCorpus,
    getCorpus,
  };
}
```

- [ ] **Step 2: Verify build**

```bash
cd frontend && npm run build
```

Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add frontend/src/hooks/useCorpora.ts
git commit -m "feat: add useCorpora hook for corpus CRUD operations"
```

---

### Task 7: Home Page (Dashboard)

**Files:**
- Modify: `frontend/src/pages/HomePage.tsx`

**Interfaces:**
- Consumes: `useCorpora()` hook
- Consumes: `<CorpusCard />` component
- Consumes: `db.savedQueries.countAll()`

- [ ] **Step 1: Implement HomePage**

Replace `frontend/src/pages/HomePage.tsx`:

```typescript
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { CorpusCard } from "../components/CorpusCard";
import { useCorpora } from "../hooks/useCorpora";
import { db } from "../lib/storage";

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
        <span className="stat-sep">·</span>
        <span className="stat">{favorites.length} favorites</span>
        <span className="stat-sep">·</span>
        <span className="stat">{queryCount} saved queries</span>
      </section>

      <section className="quick-start">
        <h2 className="section-title">Quick Start</h2>
        <div className="quick-actions">
          <button className="btn-quick" onClick={() => handleOpenCorpus("demo")}>
            Try Demo Corpus
          </button>
          <button className="btn-quick" onClick={() => navigate("/library")}>
            Upload Files
          </button>
          <button className="btn-quick-link" onClick={() => navigate("/library")}>
            View All →
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

- [ ] **Step 2: Add HomePage styles**

Add to `frontend/src/App.v2.css`:

```css
/* ============================================
   Home Page (Dashboard)
   ============================================ */

.dashboard-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--s4);
  margin-bottom: var(--s4);
}

.dashboard-section {
  background: var(--bg-primary);
  border-radius: var(--r2);
  padding: var(--s4);
}

.section-title {
  font-size: 11px;
  font-weight: 600;
  color: var(--label-tertiary);
  text-transform: uppercase;
  letter-spacing: 0.03em;
  margin-bottom: var(--s3);
}

.empty-hint {
  font-size: 14px;
  color: var(--label-tertiary);
  font-style: italic;
}

.favorites-row {
  display: flex;
  gap: var(--s3);
  overflow-x: auto;
  padding-bottom: var(--s2);
}

.recent-list {
  list-style: none;
}

.recent-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--s2) 0;
  border-bottom: 1px solid var(--separator);
  cursor: pointer;
  transition: background 0.15s ease;
  margin: 0 calc(-1 * var(--s3));
  padding-left: var(--s3);
  padding-right: var(--s3);
}

.recent-item:last-child {
  border-bottom: none;
}

.recent-item:hover {
  background: var(--fill-tertiary);
}

.recent-name {
  font-size: 14px;
  font-weight: 500;
  color: var(--label);
}

.recent-time {
  font-size: 12px;
  color: var(--label-tertiary);
}

.stats-row {
  display: flex;
  align-items: center;
  gap: var(--s2);
  padding: var(--s3) var(--s4);
  background: var(--bg-primary);
  border-radius: var(--r2);
  margin-bottom: var(--s4);
}

.stat {
  font-size: 14px;
  color: var(--label-secondary);
}

.stat-sep {
  color: var(--label-quaternary);
}

.quick-start {
  background: var(--bg-primary);
  border-radius: var(--r2);
  padding: var(--s4);
}

.quick-actions {
  display: flex;
  gap: var(--s3);
}

.btn-quick {
  padding: var(--s3) var(--s4);
  border: 1px solid var(--separator-opaque);
  background: var(--bg-primary);
  color: var(--label);
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  border-radius: var(--r2);
  transition: background 0.15s ease, border-color 0.15s ease;
}

.btn-quick:hover {
  background: var(--fill-tertiary);
  border-color: var(--system-blue);
}

.btn-quick-link {
  padding: var(--s3) var(--s4);
  border: none;
  background: none;
  color: var(--system-blue);
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
}

.btn-quick-link:hover {
  text-decoration: underline;
}

.loading-state {
  text-align: center;
  padding: var(--s6);
  color: var(--label-tertiary);
}

@media (max-width: 768px) {
  .dashboard-grid {
    grid-template-columns: 1fr;
  }

  .quick-actions {
    flex-direction: column;
  }
}
```

- [ ] **Step 3: Verify app runs**

```bash
cd frontend && npm run dev
```

Open http://localhost:5173 - should see dashboard with favorites, recent, stats, and quick actions.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/HomePage.tsx frontend/src/App.v2.css
git commit -m "feat: implement Home page dashboard with favorites, recent, and stats"
```

---

### Task 8: Library Page with Create Modal

**Files:**
- Modify: `frontend/src/pages/LibraryPage.tsx`
- Create: `frontend/src/components/CreateCorpusModal.tsx`

**Interfaces:**
- Consumes: `useCorpora()` hook
- Consumes: `<CorpusCard />` component
- Consumes: `api.ingest()` from existing API
- Consumes: `slugify()` from lib/slugify.ts
- Produces: Full library page with filtering, sorting, and create modal

- [ ] **Step 1: Create CreateCorpusModal component**

Create `frontend/src/components/CreateCorpusModal.tsx`:

```typescript
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
          <button className="modal-close" onClick={onClose}>×</button>
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
                    <button className="file-remove" onClick={() => removeFile(index)}>×</button>
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
```

- [ ] **Step 2: Implement LibraryPage**

Replace `frontend/src/pages/LibraryPage.tsx`:

```typescript
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
```

- [ ] **Step 3: Add modal and library styles**

Add to `frontend/src/App.v2.css`:

```css
/* ============================================
   Library Page
   ============================================ */

.library-toolbar {
  display: flex;
  gap: var(--s3);
  margin-bottom: var(--s4);
}

.search-input {
  flex: 1;
  padding: var(--s2) var(--s3);
  border: 1px solid var(--separator-opaque);
  background: var(--bg-primary);
  color: var(--label);
  font-size: 14px;
  border-radius: var(--r1);
  outline: none;
  transition: border-color 0.15s ease;
}

.search-input:focus {
  border-color: var(--system-blue);
}

.toolbar-select {
  padding: var(--s2) var(--s3);
  border: 1px solid var(--separator-opaque);
  background: var(--bg-primary);
  color: var(--label);
  font-size: 14px;
  border-radius: var(--r1);
  cursor: pointer;
}

.corpus-list {
  display: flex;
  flex-direction: column;
  gap: var(--s3);
}

.empty-state {
  text-align: center;
  padding: var(--s6);
  color: var(--label-tertiary);
}

.empty-state p {
  margin-bottom: var(--s3);
}

/* ============================================
   Modal
   ============================================ */

.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: var(--s4);
}

.modal-content {
  background: var(--bg-primary);
  border-radius: var(--r3);
  width: 100%;
  max-width: 500px;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

.modal-small {
  max-width: 400px;
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--s4);
  border-bottom: 1px solid var(--separator-opaque);
}

.modal-title {
  font-size: 17px;
  font-weight: 600;
  color: var(--label);
}

.modal-close {
  width: 28px;
  height: 28px;
  border: none;
  background: var(--fill);
  color: var(--label-secondary);
  font-size: 18px;
  cursor: pointer;
  border-radius: var(--r1);
  display: flex;
  align-items: center;
  justify-content: center;
}

.modal-close:hover {
  background: var(--fill-secondary);
}

.modal-body {
  padding: var(--s4);
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: var(--s3);
  padding: var(--s4);
  border-top: 1px solid var(--separator-opaque);
}

/* ============================================
   Form Elements
   ============================================ */

.form-field {
  margin-bottom: var(--s4);
}

.form-label {
  display: block;
  font-size: 13px;
  font-weight: 500;
  color: var(--label);
  margin-bottom: var(--s2);
}

.form-input {
  width: 100%;
  padding: var(--s2) var(--s3);
  border: 1px solid var(--separator-opaque);
  background: var(--bg-secondary);
  color: var(--label);
  font-size: 15px;
  border-radius: var(--r1);
  outline: none;
  transition: border-color 0.15s ease;
}

.form-input:focus {
  border-color: var(--system-blue);
  background: var(--bg-primary);
}

.form-textarea {
  width: 100%;
  padding: var(--s2) var(--s3);
  border: 1px solid var(--separator-opaque);
  background: var(--bg-secondary);
  color: var(--label);
  font-size: 15px;
  border-radius: var(--r1);
  outline: none;
  resize: vertical;
  font-family: inherit;
  transition: border-color 0.15s ease;
}

.form-textarea:focus {
  border-color: var(--system-blue);
  background: var(--bg-primary);
}

.form-error {
  color: var(--system-red);
  font-size: 13px;
  margin-top: var(--s2);
}

.file-dropzone {
  border: 2px dashed var(--separator-opaque);
  border-radius: var(--r2);
  padding: var(--s5);
  text-align: center;
  cursor: pointer;
  transition: border-color 0.15s ease, background 0.15s ease;
}

.file-dropzone:hover,
.file-dropzone.dragging {
  border-color: var(--system-blue);
  background: rgba(0, 122, 255, 0.04);
}

.dropzone-text {
  display: block;
  font-size: 14px;
  color: var(--label-secondary);
  margin-bottom: var(--s1);
}

.dropzone-hint {
  display: block;
  font-size: 12px;
  color: var(--label-tertiary);
}

.file-list {
  list-style: none;
  margin-top: var(--s3);
}

.file-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--s2);
  background: var(--fill-tertiary);
  border-radius: var(--r1);
  margin-bottom: var(--s1);
}

.file-name {
  font-size: 13px;
  color: var(--label);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.file-remove {
  background: none;
  border: none;
  color: var(--label-tertiary);
  font-size: 16px;
  cursor: pointer;
  padding: 0 var(--s1);
}

.file-remove:hover {
  color: var(--system-red);
}
```

- [ ] **Step 4: Verify app runs**

```bash
cd frontend && npm run dev
```

Open http://localhost:5173/library - should see library with demo corpus, can create new corpus.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/LibraryPage.tsx frontend/src/components/CreateCorpusModal.tsx frontend/src/App.v2.css
git commit -m "feat: implement Library page with corpus list, filtering, and create modal"
```

---

### Task 9: Search Page Integration

**Files:**
- Modify: `frontend/src/pages/SearchPage.tsx`
- Modify: `frontend/src/hooks/useDashboard.ts`

**Interfaces:**
- Consumes: Route param `corpusId`
- Consumes: `db.corpora.get(corpusId)`, `db.preferences.set("lastCorpusId", corpusId)`
- Produces: Search page with corpus context header

- [ ] **Step 1: Update useDashboard to accept corpusId**

Modify `frontend/src/hooks/useDashboard.ts` - add corpusId parameter and update lastUsedAt:

At the top of the file, add import:
```typescript
import { db } from "../lib/storage";
```

Change the function signature from:
```typescript
export function useDashboard(seedQuery: string) {
```
to:
```typescript
export function useDashboard(seedQuery: string, corpusId: string = "demo") {
```

Inside the `runQuery` callback, after `setHasRun(true);`, add:
```typescript
      // Update lastUsedAt for this corpus
      db.corpora.get(corpusId).then((corpus) => {
        if (corpus) {
          db.corpora.put({ ...corpus, lastUsedAt: Date.now() });
        }
      });
      db.preferences.set("lastCorpusId", corpusId);
```

- [ ] **Step 2: Implement SearchPage with corpus context**

Replace `frontend/src/pages/SearchPage.tsx`:

```typescript
import { useEffect, useState, FormEvent, useRef, PointerEvent as ReactPointerEvent, ChangeEvent, DragEvent } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useDashboard, type LatencyKind } from "../hooks/useDashboard";
import { db } from "../lib/storage";
import type { Corpus, Chip, FacetBucket, HistogramBin } from "../lib/types";
import type { CachedScore } from "../lib/scoreCache";

const DEFAULT_QUERY = "every place we retry a network call without backoff";

type MainTab = "results" | "facets" | "analytics";

export function SearchPage() {
  const { corpusId } = useParams<{ corpusId: string }>();
  const navigate = useNavigate();
  const [corpus, setCorpus] = useState<Corpus | null>(null);
  const [loading, setLoading] = useState(true);
  const d = useDashboard(DEFAULT_QUERY, corpusId || "demo");
  const [activeTab, setActiveTab] = useState<MainTab>("results");

  useEffect(() => {
    if (!corpusId) {
      navigate("/library");
      return;
    }
    db.corpora.get(corpusId).then((c) => {
      if (!c) {
        navigate("/library");
        return;
      }
      setCorpus(c);
      setLoading(false);
    });
  }, [corpusId, navigate]);

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void d.runQuery(d.predicate);
  };

  const toggleFavorite = async () => {
    if (!corpus || corpus.isDemo) return;
    const updated = { ...corpus, isFavorite: !corpus.isFavorite };
    await db.corpora.put(updated);
    setCorpus(updated);
  };

  if (loading || !corpus) {
    return (
      <main className="page-content">
        <div className="loading-state">Loading corpus...</div>
      </main>
    );
  }

  return (
    <div className="search-page">
      <CorpusHeader corpus={corpus} onToggleFavorite={toggleFavorite} />
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

function CorpusHeader({ corpus, onToggleFavorite }: { corpus: Corpus; onToggleFavorite: () => void }) {
  return (
    <div className="corpus-header">
      <div className="corpus-header-main">
        <h1 className="corpus-header-title">{corpus.name}</h1>
        <button
          className={`star-btn${corpus.isFavorite ? " active" : ""}`}
          onClick={onToggleFavorite}
          disabled={corpus.isDemo}
          aria-label={corpus.isFavorite ? "Remove from favorites" : "Add to favorites"}
        >
          {corpus.isFavorite ? "★" : "☆"}
        </button>
      </div>
      <div className="corpus-header-meta">
        <span>{corpus.documentCount} docs</span>
        {corpus.tags.length > 0 && (
          <>
            <span className="meta-sep">·</span>
            {corpus.tags.map((tag) => (
              <span key={tag} className="corpus-tag">#{tag}</span>
            ))}
          </>
        )}
      </div>
    </div>
  );
}

// Copy remaining components from App.v2.tsx: QueryBar, ThresholdControl, FilterBar, TabbedContent, etc.
// For brevity, import them or copy the implementations from App.v2.tsx

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
          placeholder="Describe what you're looking for in plain English..."
        />
        <button className="btn-primary" type="submit" disabled={streaming}>
          {streaming ? "Scanning..." : "Scan"}
        </button>
      </form>
    </section>
  );
}

// ThresholdControl, FilterBar, TabbedContent, ResultsPanel, AnalyticsPanel, FacetGroup, Sparkline
// These are copied from App.v2.tsx - for space, showing abbreviated versions

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
        onPointerDown={(e) => { if (hasRun) { dragging.current = true; e.currentTarget.setPointerCapture(e.pointerId); setFromClientX(e.clientX); } }}
        onPointerMove={(e) => dragging.current && setFromClientX(e.clientX)}
        onPointerUp={() => (dragging.current = false)}
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
        <div className="threshold-thumb" style={{ left: `${threshold * 100}%` }} />
        <div className="histogram-axis"><span>0</span><span>1</span></div>
      </div>
    </section>
  );
}

interface FilterBarProps {
  chips: Chip[];
  refining: boolean;
  onRefine: (utterance: string) => Promise<void>;
  onRemoveChip: (clauseId: string) => Promise<void>;
  onFreshFiles: (files: FileList | File[]) => Promise<void>;
}

function FilterBar({ chips, refining, onRefine, onRemoveChip, onFreshFiles }: FilterBarProps) {
  const [input, setInput] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (input.trim() && !refining) {
      onRefine(input.trim());
      setInput("");
    }
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files.length) onFreshFiles(e.dataTransfer.files);
  };

  return (
    <section className="filter-section">
      <div className="filter-bar">
        <div className="chip-rail">
          {chips.map((chip) => (
            <button key={chip.clause_id} className="chip" onClick={() => onRemoveChip(chip.clause_id)}>
              <span className="chip-label">{chip.label}</span>
              <span className="chip-text">{chip.text}</span>
              {chip.removable && <span className="chip-remove">×</span>}
            </button>
          ))}
          <form className="add-filter-form" onSubmit={handleSubmit}>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Refine: 'only in tests', 'exclude logging'..."
              disabled={refining}
            />
            <button type="submit" className="btn-add" disabled={!input.trim() || refining}>
              {refining ? "..." : "Add"}
            </button>
          </form>
          <label
            className={`dropzone${dragActive ? " active" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
          >
            <input ref={fileRef} type="file" multiple onChange={(e) => e.target.files && onFreshFiles(e.target.files)} />
            + Files
          </label>
        </div>
      </div>
    </section>
  );
}

// Simplified TabbedContent - full implementation would copy from App.v2.tsx
function TabbedContent(props: {
  activeTab: MainTab;
  onTabChange: (tab: MainTab) => void;
  results: CachedScore[];
  threshold: number;
  hasRun: boolean;
  streaming: boolean;
  onClickRefine: (chunkId: string, sign: "+" | "-") => Promise<void>;
  facets: { type: FacetBucket[]; category: FacetBucket[] };
  docsPerSec: number;
  elapsedMs: number;
  etaMs: number;
  latencyMs: number;
  latencyKind: LatencyKind;
  latHistory: number[];
}) {
  return (
    <section className="tabbed-section">
      <div className="tab-bar">
        <button className={`tab-btn${props.activeTab === "results" ? " active" : ""}`} onClick={() => props.onTabChange("results")}>
          Results <span className="tab-count">{props.results.length}</span>
        </button>
        <button className={`tab-btn${props.activeTab === "facets" ? " active" : ""}`} onClick={() => props.onTabChange("facets")}>
          Breakdown
        </button>
        <button className={`tab-btn${props.activeTab === "analytics" ? " active" : ""}`} onClick={() => props.onTabChange("analytics")}>
          Analytics
        </button>
      </div>
      <div className="tab-panel">
        {props.activeTab === "results" && (
          <div className="results-list">
            {props.results.map((result) => (
              <article key={result.chunk_id} className={`result-row${result.score >= props.threshold ? " matched" : ""}`}>
                <div className="result-score">
                  <span className="score-value">{result.score.toFixed(2)}</span>
                </div>
                <div className="result-content">
                  <div className="result-title">{result.meta.title}</div>
                  <div className="result-meta">{result.meta.category}</div>
                </div>
                <div className="result-actions">
                  <button className="action-btn positive" onClick={() => props.onClickRefine(result.chunk_id, "+")}>+</button>
                  <button className="action-btn negative" onClick={() => props.onClickRefine(result.chunk_id, "-")}>−</button>
                </div>
              </article>
            ))}
          </div>
        )}
        {props.activeTab === "facets" && <div className="panel-empty">Facets view</div>}
        {props.activeTab === "analytics" && <div className="panel-empty">Analytics view</div>}
      </div>
    </section>
  );
}
```

- [ ] **Step 3: Add corpus header styles**

Add to `frontend/src/App.v2.css`:

```css
/* ============================================
   Search Page
   ============================================ */

.search-page {
  display: flex;
  flex-direction: column;
  flex: 1;
}

.corpus-header {
  background: var(--bg-primary);
  border-bottom: 1px solid var(--separator-opaque);
  padding: var(--s3) var(--s4);
}

.corpus-header-main {
  display: flex;
  align-items: center;
  gap: var(--s2);
}

.corpus-header-title {
  font-size: 20px;
  font-weight: 600;
  color: var(--label);
}

.corpus-header-meta {
  display: flex;
  align-items: center;
  gap: var(--s2);
  margin-top: var(--s1);
  font-size: 13px;
  color: var(--label-tertiary);
}

.meta-sep {
  color: var(--label-quaternary);
}
```

- [ ] **Step 4: Verify app runs**

```bash
cd frontend && npm run dev
```

Navigate to http://localhost:5173/search/demo - should see search page with corpus header.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/SearchPage.tsx frontend/src/hooks/useDashboard.ts frontend/src/App.v2.css
git commit -m "feat: integrate Search page with corpus context and lastUsedAt tracking"
```

---

### Task 10: Hint Component and useHints Hook

**Files:**
- Create: `frontend/src/components/Hint.tsx`
- Create: `frontend/src/hooks/useHints.ts`

**Interfaces:**
- Consumes: `db.hints.isDismissed(key)`, `db.hints.dismiss(key)`
- Produces: `<Hint hintKey="..." message="..." />` component
- Produces: `useHints(keys)` hook

- [ ] **Step 1: Create useHints hook**

Create `frontend/src/hooks/useHints.ts`:

```typescript
import { useCallback, useEffect, useState } from "react";
import { db } from "../lib/storage";

export function useHints(keys: string[]) {
  const [dismissed, setDismissed] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all(
      keys.map(async (key) => {
        const isDismissed = await db.hints.isDismissed(key);
        return [key, isDismissed] as const;
      })
    ).then((results) => {
      setDismissed(Object.fromEntries(results));
      setLoading(false);
    });
  }, [keys.join(",")]);

  const dismiss = useCallback(async (key: string) => {
    await db.hints.dismiss(key);
    setDismissed((prev) => ({ ...prev, [key]: true }));
  }, []);

  const isVisible = useCallback(
    (key: string) => !loading && !dismissed[key],
    [loading, dismissed]
  );

  return { isVisible, dismiss, loading };
}
```

- [ ] **Step 2: Create Hint component**

Create `frontend/src/components/Hint.tsx`:

```typescript
interface HintProps {
  hintKey: string;
  message: string;
  visible: boolean;
  onDismiss: () => void;
}

export function Hint({ hintKey, message, visible, onDismiss }: HintProps) {
  if (!visible) return null;

  return (
    <div className="hint" data-hint-key={hintKey}>
      <svg className="hint-icon" viewBox="0 0 20 20" fill="currentColor" width="16" height="16">
        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z" clipRule="evenodd" />
      </svg>
      <span className="hint-text">{message}</span>
      <button className="hint-dismiss" onClick={onDismiss} aria-label="Dismiss hint">
        ×
      </button>
    </div>
  );
}
```

- [ ] **Step 3: Add Hint styles**

Add to `frontend/src/App.v2.css`:

```css
/* ============================================
   Contextual Hints
   ============================================ */

.hint {
  display: flex;
  align-items: center;
  gap: var(--s2);
  padding: var(--s2) var(--s3);
  background: rgba(0, 122, 255, 0.08);
  border: 1px solid rgba(0, 122, 255, 0.2);
  border-radius: var(--r1);
  margin-bottom: var(--s3);
}

.hint-icon {
  color: var(--system-blue);
  flex-shrink: 0;
}

.hint-text {
  flex: 1;
  font-size: 13px;
  color: var(--label-secondary);
}

.hint-dismiss {
  background: none;
  border: none;
  color: var(--label-tertiary);
  font-size: 16px;
  cursor: pointer;
  padding: 0 var(--s1);
  line-height: 1;
}

.hint-dismiss:hover {
  color: var(--label);
}
```

- [ ] **Step 4: Verify build**

```bash
cd frontend && npm run build
```

Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/Hint.tsx frontend/src/hooks/useHints.ts frontend/src/App.v2.css
git commit -m "feat: add Hint component and useHints hook for contextual onboarding"
```

---

### Task 11: Final Integration and Testing

**Files:**
- Modify: `frontend/src/pages/HomePage.tsx` (add hints)
- Modify: `frontend/src/pages/LibraryPage.tsx` (add hints)

**Interfaces:**
- Consumes: `<Hint />` component
- Consumes: `useHints()` hook

- [ ] **Step 1: Add hints to HomePage**

In `frontend/src/pages/HomePage.tsx`, add import:
```typescript
import { Hint } from "../components/Hint";
import { useHints } from "../hooks/useHints";
```

Add at the start of the component:
```typescript
  const { isVisible, dismiss } = useHints(["home-favorites-empty", "home-demo-hint"]);
```

Replace the empty favorites hint with:
```typescript
{favorites.length === 0 ? (
  <>
    <Hint
      hintKey="home-favorites-empty"
      message="Star your frequently used corpora to pin them here"
      visible={isVisible("home-favorites-empty")}
      onDismiss={() => dismiss("home-favorites-empty")}
    />
    <p className="empty-hint">No favorites yet</p>
  </>
) : (
```

- [ ] **Step 2: Add hints to LibraryPage**

In `frontend/src/pages/LibraryPage.tsx`, add import:
```typescript
import { Hint } from "../components/Hint";
import { useHints } from "../hooks/useHints";
```

Add at the start of the component:
```typescript
  const { isVisible, dismiss } = useHints(["library-empty", "library-dropzone"]);
```

In the empty state, add:
```typescript
{filteredCorpora.length === 0 ? (
  <div className="empty-state">
    <Hint
      hintKey="library-empty"
      message="Create your first corpus by uploading files or try the built-in demo"
      visible={isVisible("library-empty") && corpora.length === 0}
      onDismiss={() => dismiss("library-empty")}
    />
    <p>No corpora found</p>
```

- [ ] **Step 3: Run full test suite**

```bash
cd frontend && npm test
```

Expected: All tests pass

- [ ] **Step 4: Run build and verify**

```bash
cd frontend && npm run build
```

Expected: Build succeeds

- [ ] **Step 5: Manual testing checklist**

Run `npm run dev` and verify:
- [ ] Home page shows favorites, recent, stats
- [ ] Can navigate to Library via tab
- [ ] Can create new corpus with file upload
- [ ] New corpus appears in library and recent
- [ ] Can open corpus and search
- [ ] Search tab remembers last corpus
- [ ] Can star/unstar corpora
- [ ] Can delete non-demo corpora
- [ ] Hints appear and can be dismissed
- [ ] Demo corpus cannot be edited/deleted

- [ ] **Step 6: Final commit**

```bash
git add .
git commit -m "feat: complete corpus library with hints and full integration"
```

---

## Summary

This plan implements:
1. **Dependencies**: react-router-dom, idb
2. **Storage**: IndexedDB with corpora, savedQueries, hints, preferences
3. **Navigation**: NavBar with Home/Library/Search tabs
4. **Pages**: Home (dashboard), Library (CRUD), Search (with corpus context)
5. **Components**: CorpusCard, CreateCorpusModal, Hint
6. **Hooks**: useCorpora, useHints

Total tasks: 11
Estimated implementation: Sequential execution with commits after each task
