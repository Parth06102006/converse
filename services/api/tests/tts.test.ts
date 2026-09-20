import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import type { Server } from "node:http";
import { createApp } from "../src/app.js";

describe("Converse API - Speech TTS Endpoints", () => {
  const app = createApp();
  let server: Server;
  let baseUrl: string;

  before(async () => {
    await new Promise<void>((resolve) => {
      server = app.listen(0, "127.0.0.1", () => {
        const addr = server.address();
        if (addr && typeof addr === "object") {
          baseUrl = `http://127.0.0.1:${addr.port}`;
        }
        resolve();
      });
    });
  });

  after(async () => {
    await new Promise<void>((resolve, reject) => {
      server.close((err) => {
        if (err) reject(err);
        else resolve();
      });
    });
  });

  it("exposes /api/speech/tts in /api/info endpoints catalog", async () => {
    const res = await fetch(`${baseUrl}/api/info`);
    assert.equal(res.status, 200);
    const body = (await res.json()) as { endpoints: Array<{ path: string; method: string }> };
    const ttsEndpoint = body.endpoints.find(
      (e) => e.path === "/api/speech/tts" && e.method === "POST",
    );
    assert.ok(ttsEndpoint, "Expected /api/speech/tts in endpoints list");
  });

  it("returns 400 when text payload is empty", async () => {
    const res = await fetch(`${baseUrl}/api/speech/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "   " }),
    });
    assert.equal(res.status, 400);
    const body = (await res.json()) as { ok: boolean; error: { code: string } };
    assert.equal(body.ok, false);
    assert.equal(body.error.code, "EMPTY_TEXT_PAYLOAD");
  });
});
