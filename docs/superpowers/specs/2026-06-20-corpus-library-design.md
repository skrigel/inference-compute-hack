# Corpus Library & Dashboard Design

**Date:** 2026-06-20
**Status:** Approved
**Scope:** Frontend multi-page app with corpus management, dashboard, and persistent storage

---

## Overview

Transform the single-page search UI into a multi-page application where researchers/analysts can:
- Manage multiple corpora with rich metadata
- Access a dashboard with favorites, recent corpora, and stats
- Save and restore queries within each corpus
- Navigate via persistent top-level tabs

**Target users:** Power users (researchers/analysts) who return to multiple corpora repeatedly.

**Storage:** Browser-only (IndexedDB) to start. Backend persistence deferred.

**Design system:** Apple Human Interface Guidelines — consistent with existing App.v2.css styling (SF fonts, system colors, 8pt spacing grid).

---

## 1. Navigation & Routing

### Routes

| Path | Page | Description |
|------|------|-------------|
| `/` | Home | Dashboard with favorites, recent, stats, quick actions |
| `/library` | Library | Full corpus list with create, edit, delete, filtering |
| `/search/:corpusId` | Search | Semantic search interface for a specific corpus |

### Top Navigation Bar

```
┌──────────────────────────────────────────────────────────┐
│  grepmeaning    [Home]  [Library]  [Search]      (?) (⚙)│
└──────────────────────────────────────────────────────────┘
```

- Active tab highlighted with system blue
- "Search" tab disabled (grayed) if no corpus selected
- Search tab remembers last-used corpus (via `localStorage.lastCorpusId`)

### Routing Behavior

- First visit → redirects to `/` (Home)
- Clicking corpus card → navigates to `/search/:corpusId`
- Unknown routes → redirect to Home

---

## 2. Data Model

### Corpus Record

```typescript
interface Corpus {
  id: string;              // URL-safe slug, e.g. "my-research-2024"
                           // Generated: name.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 50)
  name: string;            // Display name
  description: string;     // User notes about this corpus
  tags: string[];          // For filtering/organizing
  createdAt: number;       // Unix timestamp (ms)
  lastUsedAt: number;      // Updated on each search session
  isFavorite: boolean;     // Star/pin to top
  isDemo: boolean;         // True for built-in demo corpus (read-only)
  documentCount: number;   // Cached count from last ingest
  source: "files" | "demo";// How it was created
}
```

### Saved Query Record

```typescript
interface SavedQuery {
  id: string;              // UUID
  corpusId: string;        // Foreign key to Corpus.id
  predicate: string;       // The search query text
  threshold: number;       // Threshold at time of save
  chips: Chip[];           // Refinement chips (from existing types.ts)
  name: string;            // User-provided name
  notes: string;           // User annotations
  savedAt: number;         // Unix timestamp (ms)
}
```

### Demo Corpus

Pre-seeded on first load:
- `id: "demo"`
- `name: "Demo Corpus"`
- `description: "Sample papers and code to explore the tool"`
- `isDemo: true`
- Cannot be deleted, renamed, or have files added

---

## 3. Home Page (Dashboard)

### Purpose

Quick access to frequently used corpora and high-level overview.

### Layout

```
┌──────────────────────────────────────────────────────────┐
│  Welcome back                              [+ New Corpus]│
├────────────────────────┬─────────────────────────────────┤
│  FAVORITES             │   RECENT                        │
│  ┌─────┐ ┌─────┐       │   • Research Papers  (2h ago)   │
│  │ ★   │ │ ★   │       │   • Code Review Set  (yesterday)│
│  │Corp1│ │Corp2│       │   • Demo Corpus      (3d ago)   │
│  └─────┘ └─────┘       │                                 │
├────────────────────────┴─────────────────────────────────┤
│  STATS                                                   │
│  3 corpora  ·  2 favorites  ·  12 saved queries          │
├──────────────────────────────────────────────────────────┤
│  QUICK START                                             │
│  [Try Demo Corpus]   [Upload Files]   [View All →]       │
└──────────────────────────────────────────────────────────┘
```

### Components

- **Favorites section:** Horizontal row of corpus cards (max 4-5 visible, scrollable)
- **Recent section:** Vertical list sorted by `lastUsedAt` desc (max 5)
- **Stats row:** Simple counts pulled from IndexedDB
- **Quick start:** Action buttons for common flows

### Behavior

- Clicking corpus → `/search/:corpusId`
- "+ New Corpus" → opens create modal (or navigates to Library with create open)
- "View All →" → `/library`
- Empty favorites → hint text "Star corpora to pin them here"

---

## 4. Library Page

### Purpose

Full corpus management — browse, search, filter, create, edit, delete.

### Layout

```
┌──────────────────────────────────────────────────────────┐
│  Corpus Library                          [+ New Corpus]  │
├──────────────────────────────────────────────────────────┤
│  [Search corpora...]     [Filter ▾]  [Sort: Recent ▾]   │
├──────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────┐  │
│  │ ★ Research Papers                        12 docs   │  │
│  │   ML and NLP papers from 2023-2024                 │  │
│  │   #research #ml          Last used: 2h ago         │  │
│  │                              [Open] [Edit] [Delete]│  │
│  └────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────┐  │
│  │   Code Review Set                        847 docs  │  │
│  │   Internal codebase for security audit             │  │
│  │   #code #security        Last used: yesterday      │  │
│  │                              [Open] [Edit] [Delete]│  │
│  └────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────┐  │
│  │   Demo Corpus (built-in)                  24 docs  │  │
│  │   Sample papers and code to explore the tool       │  │
│  │                          Last used: 3d ago         │  │
│  │                              [Open]                │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Features

| Feature | Description |
|---------|-------------|
| Search | Filter corpora by name/description text match |
| Filter dropdown | By tag, by favorite only, by source (files/demo) |
| Sort dropdown | Recent (lastUsedAt), Alphabetical (name), Size (documentCount) |
| Star toggle | Click star icon to favorite/unfavorite |
| Open | Navigate to `/search/:corpusId` |
| Edit | Modal to update name, description, tags |
| Delete | Confirmation dialog; not available for demo corpus |

### Create Corpus Flow

1. Click "+ New Corpus" → Modal opens
2. Fields:
   - Name (required, text input)
   - Description (optional, textarea)
   - Tags (optional, comma-separated or tag input)
   - File drop zone (required, accepts multiple files)
3. File types: `.txt`, `.md`, `.pdf`, `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.go`, `.rs`, `.java`, `.c`, `.cpp`, `.h`
4. Submit: "Create Corpus"
   - Validates name is non-empty and unique
   - Generates URL-safe `id` from name
   - Calls `api.ingest(corpusId, documents)`
   - Saves corpus metadata to IndexedDB
   - Navigates to `/search/:newCorpusId`

---

## 5. Search Page Updates

### Header Integration

```
┌──────────────────────────────────────────────────────────┐
│  grepmeaning    [Home]  [Library]  [Search]      (?) (⚙)│
├──────────────────────────────────────────────────────────┤
│  Research Papers  ★                    [Save Query] [↗] │
│  12 docs · #research #ml                                 │
└──────────────────────────────────────────────────────────┘
```

- Corpus name as subtitle with inline star toggle
- Document count and tags displayed
- **Save Query** button: saves current `{predicate, threshold, chips}` to IndexedDB
- **↗** link: navigates to Library with edit modal open for this corpus

### Saved Queries Drawer

Collapsible section below the filter bar:

```
┌─────────────────────────────────┐
│  Saved Queries (3)          [−]│
├─────────────────────────────────┤
│  • Retry without backoff        │
│  • Authentication flows         │
│  • Error handling patterns      │
└─────────────────────────────────┘
```

- Click query → loads predicate, threshold, chips into search state
- Hover → shows notes tooltip if present
- Context menu or icon → delete saved query

### Route Integration

- Route: `/search/:corpusId`
- On mount:
  1. Read corpus metadata from IndexedDB (for display)
  2. Call `api.ingest(corpusId)` to warm/load the corpus
  3. Update `lastUsedAt` in IndexedDB
- If corpus not found → redirect to `/library` with error toast

---

## 6. Contextual Hints (Onboarding)

### Strategy

Inline hints that appear contextually, dismissible, and don't reappear once dismissed.

### Hint Locations

| Location | Condition | Hint Text |
|----------|-----------|-----------|
| Home - Favorites | Empty favorites | "Star your frequently used corpora to pin them here" |
| Home - Demo card | First visit | "Try the demo corpus to see how semantic search works" |
| Library - Empty | No corpora | "Create your first corpus by uploading files or try the built-in demo" |
| Library - Drop zone | Always (until dismissed) | "Drag PDFs, text files, or code files here" |
| Search - Query input | Placeholder | "Describe what you're looking for in plain English..." |
| Search - Threshold | First use | "Drag to adjust relevance cutoff" |
| Search - Refine input | Placeholder | "Refine: 'only in tests', 'exclude logging'..." |
| Search - Saved queries | Empty | "Save queries you want to return to" |

### Implementation

```typescript
interface HintState {
  [key: string]: boolean; // true = dismissed
}

function useHints() {
  // Reads/writes to IndexedDB hints store
  // Returns { isVisible(key), dismiss(key) }
}
```

### Visual Style (Apple HIG)

- Background: `rgba(0, 122, 255, 0.08)` (subtle blue tint)
- Border: `1px solid rgba(0, 122, 255, 0.2)`
- Border radius: `var(--r1)` (6px)
- Icon: SF Symbol info.circle or lightbulb
- Text: 13px, `var(--label-secondary)`
- Dismiss: Small × button, right-aligned

---

## 7. Storage Layer (IndexedDB)

### Database

- Name: `grepmeaning-db`
- Version: `1`

### Object Stores

| Store | Key Path | Indexes |
|-------|----------|---------|
| `corpora` | `id` | `lastUsedAt`, `isFavorite`, `createdAt` |
| `savedQueries` | `id` | `corpusId`, `savedAt` |
| `hints` | `key` | — |
| `preferences` | `key` | — |

### Storage API

```typescript
// lib/storage.ts
import { openDB, DBSchema } from 'idb';

interface GrepMeaningDB extends DBSchema {
  corpora: {
    key: string;
    value: Corpus;
    indexes: {
      'by-lastUsedAt': number;
      'by-isFavorite': number; // 1 or 0
      'by-createdAt': number;
    };
  };
  savedQueries: {
    key: string;
    value: SavedQuery;
    indexes: {
      'by-corpusId': string;
      'by-savedAt': number;
    };
  };
  hints: { key: string; value: { key: string; dismissed: boolean } };
  preferences: { key: string; value: { key: string; value: unknown } };
}

export const db = {
  corpora: {
    getAll(): Promise<Corpus[]>,
    get(id: string): Promise<Corpus | undefined>,
    put(corpus: Corpus): Promise<void>,
    delete(id: string): Promise<void>,
    getFavorites(): Promise<Corpus[]>,
    getRecent(limit: number): Promise<Corpus[]>,
  },
  savedQueries: {
    getByCorpus(corpusId: string): Promise<SavedQuery[]>,
    put(query: SavedQuery): Promise<void>,
    delete(id: string): Promise<void>,
  },
  hints: {
    isDismissed(key: string): Promise<boolean>,
    dismiss(key: string): Promise<void>,
  },
  preferences: {
    get<T>(key: string): Promise<T | undefined>,
    set<T>(key: string, value: T): Promise<void>,
  },
};
```

### Initialization

On app load (`main.tsx` or root component):

1. Open/create database with schema
2. Check if demo corpus exists
3. If not, seed demo corpus record
4. Read `preferences.lastCorpusId` for Search tab state

### Dependencies

- `idb` package (lightweight IndexedDB wrapper with promises)

---

## 8. File Structure (New/Modified)

```
frontend/src/
├── main.tsx                    # Add BrowserRouter wrapper
├── App.tsx                     # Router outlet, nav bar
├── pages/
│   ├── HomePage.tsx            # Dashboard
│   ├── LibraryPage.tsx         # Corpus management
│   └── SearchPage.tsx          # Existing App.v2 content, adapted
├── components/
│   ├── NavBar.tsx              # Top navigation
│   ├── CorpusCard.tsx          # Reusable corpus display card
│   ├── CreateCorpusModal.tsx   # New corpus form
│   ├── EditCorpusModal.tsx     # Edit corpus metadata
│   ├── SavedQueriesDrawer.tsx  # Collapsible saved queries list
│   └── Hint.tsx                # Contextual hint component
├── hooks/
│   ├── useDashboard.ts         # Existing (minor updates)
│   ├── useCorpora.ts           # CRUD operations for corpora
│   ├── useSavedQueries.ts      # CRUD for saved queries
│   └── useHints.ts             # Hint visibility state
├── lib/
│   ├── storage.ts              # IndexedDB wrapper
│   ├── api.ts                  # Existing (no changes)
│   └── types.ts                # Add Corpus, SavedQuery types
└── App.v2.css                  # Existing + new page styles
```

---

## 9. Anti-Patterns to Avoid

- **Over-fetching:** Don't load all saved queries on Home; only load counts
- **Blocking initialization:** DB init should not block render; use suspense or loading state
- **URL mismatches:** Ensure corpus `id` is always URL-safe (slugify on create)
- **Demo corpus mutations:** Explicitly guard against edit/delete on demo corpus
- **Lost state on navigation:** Preserve search state when switching tabs (use route state or context)

---

## 10. Out of Scope (Deferred)

- Backend persistence / user accounts
- Corpus sharing between users
- Import/export corpus as file
- Full-text search within saved queries
- Undo/redo for query refinements

---

## Appendix: Apple HIG Compliance Checklist

- [ ] Touch targets ≥ 44pt
- [ ] System fonts (SF Pro, SF Mono)
- [ ] System colors (--system-blue, --system-red, etc.)
- [ ] 8pt spacing grid
- [ ] Consistent border radius (6px, 10px, 14px)
- [ ] Proper focus states with outline
- [ ] Loading states with subtle animation
- [ ] Destructive actions in red, require confirmation
- [ ] Empty states with guidance
