import { useCallback, useEffect, useState } from "react";
import { db } from "../lib/storage";

export interface Hint {
  id: string;
  title: string;
  body: string;
  context: "home" | "library" | "search" | "threshold" | "filter";
}

const ALL_HINTS: Hint[] = [
  {
    id: "welcome",
    title: "Welcome to Grep for Meaning",
    body: "Search your document corpora using natural language queries. Start by opening a corpus from the Library.",
    context: "home",
  },
  {
    id: "threshold",
    title: "Adjust the Threshold",
    body: "Drag the slider left to include more results, or right to be more selective. Only documents above the threshold are shown.",
    context: "threshold",
  },
  {
    id: "refinement",
    title: "Refine Your Search",
    body: "Add filters like 'only python' or 'without tests' to narrow results. Click chips to remove them.",
    context: "filter",
  },
  {
    id: "corpus-create",
    title: "Create a Corpus",
    body: "Upload text files, code, or documents to create a searchable corpus. Supports .txt, .md, .py, .js, and more.",
    context: "library",
  },
  {
    id: "favorites",
    title: "Star Your Favorites",
    body: "Click the star on any corpus to pin it to your Home page for quick access.",
    context: "library",
  },
];

export function useHints(context: Hint["context"]) {
  const [currentHint, setCurrentHint] = useState<Hint | null>(null);
  const [dismissedIds, setDismissedIds] = useState<Set<string>>(new Set());
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    // Load dismissed hints from storage
    async function loadDismissed() {
      const dismissed = new Set<string>();
      for (const hint of ALL_HINTS) {
        if (await db.hints.isDismissed(hint.id)) {
          dismissed.add(hint.id);
        }
      }
      setDismissedIds(dismissed);
      setLoaded(true);
    }
    loadDismissed();
  }, []);

  useEffect(() => {
    if (!loaded) return;
    // Find a hint for this context that hasn't been dismissed
    const available = ALL_HINTS.filter(
      (h) => h.context === context && !dismissedIds.has(h.id)
    );
    setCurrentHint(available[0] ?? null);
  }, [context, dismissedIds, loaded]);

  const dismiss = useCallback(async () => {
    if (!currentHint) return;
    await db.hints.dismiss(currentHint.id);
    setDismissedIds((prev) => new Set([...prev, currentHint.id]));
  }, [currentHint]);

  const dismissAll = useCallback(async () => {
    const allIds = ALL_HINTS.map((h) => h.id);
    for (const id of allIds) {
      await db.hints.dismiss(id);
    }
    setDismissedIds(new Set(allIds));
  }, []);

  return { hint: currentHint, dismiss, dismissAll };
}
