// Imports the page's entry module the way a browser does and reports whether it started.
//
// Called by `tests/contracts/test_dashboard_escapes.py` with `{site}` on stdin. The stub
// document has a `readyState`, which is what tells `main.js` it is in a real page and should
// start; the notice in `index.html` that the script did not start is `boot-status`, and
// `start()` is the only thing that hides it. Prints `{imported, error, hidden}`.
import fs from "node:fs";
import { pathToFileURL } from "node:url";

const cfg = JSON.parse(fs.readFileSync(0, "utf8"));
const notice = { hidden: false };
const els = {};
function el() {
  return {
    innerHTML: "", textContent: "", hidden: false, className: "", _c: {},
    querySelector(s) { return this._c[s] ??= el(); },
  };
}
globalThis.document = {
  readyState: "complete",
  getElementById: id => id === "boot-status" ? notice : els[id] ??= el(),
  addEventListener() {},
};
globalThis.setInterval = () => 0;
globalThis.fetch = async () => ({ ok: false });

let error = null;
try {
  await import(pathToFileURL(`${cfg.site}/js/main.js`).href);
  // `start()` kicks `render()` off without awaiting it; let it settle before reporting.
  await new Promise(r => setTimeout(r, 50));
} catch (e) {
  error = String(e?.name ?? e);
}
console.log(JSON.stringify({ imported: error === null, error, hidden: notice.hidden }));
process.exit(0);
