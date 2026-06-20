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
    // Use future timestamps to ensure they're more recent than the demo corpus
    const future = Date.now() + 10000;
    await db.corpora.put({
      id: "old", name: "Old", description: "", tags: [], createdAt: future,
      lastUsedAt: future, isFavorite: false, isDemo: false, documentCount: 1, source: "files",
    });
    await db.corpora.put({
      id: "new", name: "New", description: "", tags: [], createdAt: future + 1000,
      lastUsedAt: future + 1000, isFavorite: false, isDemo: false, documentCount: 1, source: "files",
    });
    const recent = await db.corpora.getRecent(2);
    expect(recent[0].id).toBe("new");
    expect(recent[1].id).toBe("old");
  });
});
