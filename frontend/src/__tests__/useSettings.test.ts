// @vitest-environment jsdom
import { renderHook, act } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
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
