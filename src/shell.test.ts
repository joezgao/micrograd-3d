// The static first frame in index.html must say what App.tsx says, or slow-network visitors
// see one text and then a different one when React takes over.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const norm = (s: string) => s.replace(/\s+/g, " ").trim();
const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
const app = readFileSync(new URL("./App.tsx", import.meta.url), "utf8");
const pick = (src: string, re: RegExp) => norm(src.match(re)?.[1] ?? "");

describe("static first frame", () => {
  it("has the title, lede and stage note", () => {
    expect(pick(html, /<h1>([^<]+)<\/h1>/)).toBe("micrograd 3D");
    expect(pick(html, /class="lede">([^<]+)</)).not.toBe("");
    expect(pick(html, /class="stage-note mono">([^<]+)</)).not.toBe("");
  });

  it("matches App.tsx word for word", () => {
    expect(pick(html, /<h1>([^<]+)<\/h1>/)).toBe(pick(app, /<h1>([^<]+)<\/h1>/));
    expect(pick(html, /class="lede">([^<]+)</)).toBe(pick(app, /className="lede">([^<]+)</));
    expect(pick(html, /class="stage-note mono">([^<]+)</)).toBe(pick(app, /className="stage-note mono">([^<]+)</));
  });
});
