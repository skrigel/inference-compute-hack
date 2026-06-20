# Task 2: IndexedDB Storage Layer

## Files
- Create: `frontend/src/lib/storage.ts`
- Create: `frontend/src/lib/storage.test.ts`

## Interfaces Produced
- `initDB(): Promise<void>` - initializes database and seeds demo corpus
- `clearDB(): Promise<void>` - clears database (for testing)
- `db.corpora.getAll(): Promise<Corpus[]>`
- `db.corpora.get(id: string): Promise<Corpus | undefined>`
- `db.corpora.put(corpus: Corpus): Promise<void>`
- `db.corpora.delete(id: string): Promise<void>`
- `db.corpora.getFavorites(): Promise<Corpus[]>`
- `db.corpora.getRecent(limit: number): Promise<Corpus[]>`
- `db.savedQueries.getByCorpus(corpusId: string): Promise<SavedQuery[]>`
- `db.savedQueries.put(query: SavedQuery): Promise<void>`
- `db.savedQueries.delete(id: string): Promise<void>`
- `db.savedQueries.countByCorpus(corpusId: string): Promise<number>`
- `db.savedQueries.countAll(): Promise<number>`
- `db.hints.isDismissed(key: string): Promise<boolean>`
- `db.hints.dismiss(key: string): Promise<void>`
- `db.preferences.get<T>(key: string): Promise<T | undefined>`
- `db.preferences.set<T>(key: string, value: T): Promise<void>`

## Types to Import
From `./types`: `Corpus`, `SavedQuery`

## Database Schema
- Database name: `grepmeaning-db`
- Version: `1`
- Object stores:
  - `corpora` (keyPath: `id`, indexes: `by-lastUsedAt`, `by-isFavorite`, `by-createdAt`)
  - `savedQueries` (keyPath: `id`, indexes: `by-corpusId`, `by-savedAt`)
  - `hints` (keyPath: `key`)
  - `preferences` (keyPath: `key`)

## Demo Corpus (seeded on init)
```typescript
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
```

## Test Cases Required

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

## Implementation

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

## Steps
1. Write the test file first
2. Run tests to verify they fail (module not found)
3. Write the implementation
4. Run tests to verify they pass
5. Commit with message: `feat: add IndexedDB storage layer with corpus and query persistence`

## Commands
- Run tests: `cd frontend && npm test -- storage.test.ts`
- Build: `cd frontend && npm run build`
