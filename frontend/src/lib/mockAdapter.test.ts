import { describe, expect, it } from "vitest";
import { HIST_BINS } from "./types";
import { queryMock } from "./mockAdapter";

describe("queryMock", () => {
  it("streams result, aggregate, and done events with contract-shaped payloads", async () => {
    const events = [];

    for await (const event of queryMock({
      predicate: "retry a network call without backoff",
      threshold: 0.5,
    })) {
      events.push(event);
    }

    expect(events.some((event) => event.type === "result")).toBe(true);
    expect(events.some((event) => event.type === "aggregate")).toBe(true);
    expect(events[events.length - 1]?.type).toBe("done");

    const aggregate = events.find((event) => event.type === "aggregate");
    expect(aggregate).toMatchObject({
      type: "aggregate",
      threshold: 0.5,
    });
    if (aggregate?.type !== "aggregate") {
      throw new Error("expected aggregate event");
    }
    expect(aggregate.histogram).toHaveLength(HIST_BINS);
    expect(aggregate.facets.type.length).toBeGreaterThan(0);
  });
});
