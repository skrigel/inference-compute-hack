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

const DEMO_CORPORA: Corpus[] = [
  {
    id: "demo",
    name: "Retry & Networking",
    description: "Code and papers about retry logic, backoff, and resilient networking",
    tags: ["python", "go", "networking", "distributed-systems"],
    createdAt: Date.now() - 86400000 * 2,
    lastUsedAt: Date.now(),
    isFavorite: true,
    isDemo: true,
    documentCount: 24,
    source: "demo",
  },
  {
    id: "browsecomp",
    name: "BrowseComp+",
    description: "Web documents: news, wikis, blogs, academic papers, and forums",
    tags: ["web", "retrieval", "benchmark"],
    createdAt: Date.now() - 86400000 * 5,
    lastUsedAt: Date.now() - 86400000,
    isFavorite: false,
    isDemo: true,
    documentCount: 100,
    source: "demo",
  },
  {
    id: "arxiv-ml",
    name: "arXiv ML Papers",
    description: "Machine learning papers: transformers, quantization, inference optimization",
    tags: ["ml", "transformers", "research"],
    createdAt: Date.now() - 86400000 * 3,
    lastUsedAt: Date.now() - 3600000,
    isFavorite: false,
    isDemo: true,
    documentCount: 25,
    source: "demo",
  },
  {
    id: "codebase",
    name: "Open Source Code",
    description: "Popular Python, Go, and Rust libraries: requests, grpc, tokio",
    tags: ["python", "go", "rust", "open-source"],
    createdAt: Date.now() - 86400000 * 7,
    lastUsedAt: Date.now() - 86400000 * 2,
    isFavorite: false,
    isDemo: true,
    documentCount: 25,
    source: "demo",
  },
];

export async function initDB(): Promise<void> {
  const database = await getDB();
  for (const corpus of DEMO_CORPORA) {
    const existing = await database.get("corpora", corpus.id);
    if (!existing) {
      await database.put("corpora", corpus);
    }
  }
}

export async function clearDB(): Promise<void> {
  if (dbInstance) {
    dbInstance.close();
    dbInstance = null;
  }
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
