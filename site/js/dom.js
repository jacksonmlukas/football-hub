// The one place a panel's parts touch the document.

import { panelParts } from "./state.js";

export function panel(id, opts) {
  const parts = panelParts(opts);
  const el = document.getElementById(id);
  el.querySelector(".age").textContent = parts.age;
  const body = el.querySelector(".body");
  body.className = `body ${parts.cls}`;
  body.innerHTML = parts.body;
}
