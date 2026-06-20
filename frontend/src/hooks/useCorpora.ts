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
