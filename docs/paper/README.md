# System paper draft — VoiceMOS 2026 Track 3 (Team T15)

A write-up of the PD-RAMP system: method, validation methodology, official results,
ablations and negative results. Not submitted anywhere — kept here as documentation.

## Files
- `main.tex` — the paper (~4 pages of content + references)
- `spconf.sty` — two-column US-letter conference layout

## How to compile
**Overleaf (easiest):** upload `main.tex` + `spconf.sty` into a new project,
set the compiler to pdfLaTeX, and recompile.

**Locally** (needs a TeX distribution such as MacTeX/BasicTeX):
```
pdflatex main.tex && pdflatex main.tex
```
Run twice so cross-references resolve. The bibliography is a manual
`thebibliography`, so no BibTeX pass is needed.

> Not yet compiled — no TeX distribution was available on the machine it was
> written on, so the layout is unverified.

## If you ever do submit it
1. Swap in the official style file from the target venue's author kit.
2. Check that venue's page limit and trim if needed (the negative-results
   section is the most expendable).
3. Verify the references — they were written from memory and the venue/year
   details should be checked against the actual publications.
4. Replace the ASCII-derived architecture description with the rendered
   diagram in `results/figures/architecture.png`.
