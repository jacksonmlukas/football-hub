// Runs the page's real `render()` in node against a stub DOM and a stub `fetch`, and prints
// what landed in the footer, each panel and each statement slot as JSON.
//
// Called by `tests/contracts/test_dashboard_escapes.py` with `{site, ov}` on stdin: `site` is the
// directory holding `index.html`, `js/` and `data/` (a mutated copy, for the excision
// harness) and `ov` maps an artifact name to a payload that replaces the committed one
// (`null` is a 404). The modules are imported from `site`, so a mutant is exercised
// through the same import graph the browser walks; nothing is lifted out of a file.
import fs from "node:fs";
import { pathToFileURL } from "node:url";

const cfg = JSON.parse(fs.readFileSync(0, "utf8"));

function el() {
  return {
    innerHTML: "", textContent: "", hidden: false, className: "", _c: {},
    querySelector(s) { return this._c[s] ??= el(); },
  };
}
const els = {};
globalThis.document = { getElementById: id => els[id] ??= el(), addEventListener() {} };
globalThis.setInterval = () => 0;
globalThis.fetch = async (url) => {
  const n = url.replace(/^data\//, "").replace(/\.json.*$/, "");
  if (n in cfg.ov) {
    return cfg.ov[n] === null ? { ok: false } : { ok: true, json: async () => cfg.ov[n] };
  }
  const f = `${cfg.site}/data/${n}.json`;
  return fs.existsSync(f) ? { ok: true, json: async () => JSON.parse(fs.readFileSync(f)) }
                          : { ok: false };
};

const { render } = await import(pathToFileURL(`${cfg.site}/js/main.js`).href);
await render();
const out = { footer: els.footer.innerHTML, panels: {} };
for (const [id, e] of Object.entries(els)) {
  if (id.startsWith("p-")) {
    out.panels[id] = {
      body: e._c[".body"]?.innerHTML ?? "", cls: e._c[".body"]?.className ?? "",
      age: e._c[".age"]?.textContent ?? "",
    };
  }
  // A statement slot (#354) is its own element, not a panel's child: it carries the state class.
  if (id.startsWith("st-")) out.panels[id] = { body: e.innerHTML, cls: e.className, age: "" };
}
console.log(JSON.stringify(out));
