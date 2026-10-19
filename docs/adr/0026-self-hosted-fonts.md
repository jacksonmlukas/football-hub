# Self-hosted font files are allowed under K1

**Status:** accepted 2026-10-16 (option B adopted by the maintainer, issue #477); download of the
files approved 2026-10-19. Amends K1 (#347 and the 2026-08-24 dashboard entry in
`docs/foundation-plan.md`).

**Decision.** K1 said: no webfont, no CDN, no build step. It now says: **no CDN, no build step,
no third-party request; font files may be committed to `site/fonts/` and served from the same
origin.** Every `@font-face` uses `font-display: swap` and a system-font fallback stack, so a
failed load never blanks text.

## Why option B

| | Option | Cost | Gives |
|---|---|---|---|
| A | Keep K1; system fonts only | 0 KB | Nothing changes |
| **B** | **Same-origin WOFF2 in `site/fonts/`** | **74.5 KB** | **A designed type system; still no CDN, no build, no third-party request** |
| C | Drop the no-CDN clause, load Google Fonts | 0 KB in repo | Same look, but a third-party request on every view and a render that depends on it |

The no-CDN clause exists so the page makes no request this repo cannot satisfy; B keeps that
property and C gives it up. The no-webfont clause had no stated reason beyond weight, and B
bounds the weight. The faces are the proposal, not the decision: they can change without
another ADR as long as the four properties below hold.

## What the files are

All three families are under the SIL Open Font License 1.1, which permits redistribution; each
licence text sits beside its fonts. Latin subset (U+0020-007E, U+00A0-00FF and a dozen
typographic punctuation and arrow code points), WOFF2, built with `fonttools`/`brotli` via
`uv run --with` (dev-only, not a project dependency). OpenType features kept: `tnum`, `kern`,
`liga`, `calt`, `ccmp`, `locl`, `case`, `mark`, `mkmk` (a feature a font lacks is simply absent;
Geist keeps `tnum`).

| File | Weight | Bytes | Source |
|---|---|---|---|
| `BigShouldersDisplay-Bold.woff2` | 700 | 12,040 | google/fonts `ofl/bigshouldersdisplay/` at commit 47d54ca, variable font instanced at wght=700 |
| `Geist-Regular.woff2` | 400 | 16,024 | vercel/geist-font release v1.8.1, static TTF |
| `Geist-Medium.woff2` | 500 | 16,532 | same |
| `Geist-SemiBold.woff2` | 600 | 16,488 | same |
| `GeistMono-Regular.woff2` | 400 | 13,380 | same |
| | **total** | **74,464** | |

Licences: `BigShouldersDisplay-OFL.txt` (Copyright 2019 The Big Shoulders Project Authors) and
`Geist-OFL.txt` / `GeistMono-OFL.txt` (Copyright 2024 The Geist Project Authors), each checked
to be the SIL OFL 1.1 text. Geist Mono is for hashes and digests only.

## What keeps it true

`tests/contracts/test_fonts_are_self_hosted.py`: every `@font-face` source is relative and no
page names a font host; every font file has `<Family>-OFL.txt` beside it; total font weight is
at most 120 KB; every `@font-face` says `font-display: swap`. Each check has a planted control
that must go red. `.github/workflows/pages.yml` uploads all of `site/`, so `site/fonts/` ships.

## Not decided here

How the faces are used (the visual system) is #478. This change only declares them.
