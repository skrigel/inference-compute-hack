// POST a JSON body and parse a text/event-stream response. We use
// fetch + ReadableStream (not EventSource) because /query and /refine are POSTs
// with bodies. Frames are `data: <json>\n\n`; events are routed by `.type`.
export type SseHandler = (event: unknown) => void;

export async function streamPost(
  url: string,
  body: unknown,
  onEvent: SseHandler,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`stream failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatch = (frame: string) => {
    const dataLine = frame.split("\n").find((line) => line.startsWith("data:"));
    if (!dataLine) return;
    const payload = dataLine.slice(dataLine.indexOf(":") + 1).trim();
    if (!payload) return;
    try {
      onEvent(JSON.parse(payload));
    } catch (error) {
      // A single malformed frame must not tear down the whole stream.
      console.error("dropping malformed SSE frame", payload, error);
    }
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    // Strip CR so CRLF frame terminators ("\r\n\r\n") still split on "\n\n".
    buffer += decoder.decode(value, { stream: true }).replace(/\r/g, "");
    let split: number;
    while ((split = buffer.indexOf("\n\n")) >= 0) {
      dispatch(buffer.slice(0, split));
      buffer = buffer.slice(split + 2);
    }
  }

  if (buffer.trim()) dispatch(buffer);
}
