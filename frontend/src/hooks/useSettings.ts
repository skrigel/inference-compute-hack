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
