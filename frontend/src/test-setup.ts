import "fake-indexeddb/auto";

// Polyfill localStorage for tests
if (typeof window !== "undefined") {
  const localStorageMock = {
    store: {} as Record<string, string>,
    getItem(key: string) {
      return this.store[key] || null;
    },
    setItem(key: string, value: string) {
      this.store[key] = value;
    },
    removeItem(key: string) {
      delete this.store[key];
    },
    clear() {
      this.store = {};
    },
  };

  Object.defineProperty(window, "localStorage", {
    value: localStorageMock,
    writable: true,
    configurable: true,
  });
}
