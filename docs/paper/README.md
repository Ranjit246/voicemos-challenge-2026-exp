# ICASSP 2027 paper — VoiceMOS 2026 Track 3 (Team T15)

## Files
- `main.tex` — the paper (4 pages of content + references)
- `spconf.sty` — ICASSP-style layout. **Replace with the official `spconf.sty`
  from the ICASSP 2027 author kit before submitting.**

## How to compile
**Overleaf (easiest):** upload `main.tex` + `spconf.sty` into a new project,
set the compiler to pdfLaTeX, and Recompile.

**Locally** (needs a TeX distribution — MacTeX/BasicTeX):
```
pdflatex main.tex && pdflatex main.tex
```
Run twice so cross-references resolve. The bibliography is a manual
`thebibliography`, so no BibTeX pass is needed.

## Before submitting
1. Swap in the official ICASSP 2027 `spconf.sty` (and `IEEEbib.bst` if you
   convert the bibliography to BibTeX).
2. Check the page limit: ICASSP allows 4 pages of content + 1 page of
   references only. Trim Section 6 (negative results) first if over.
3. Verify the reference details — they were written from memory and the
   volume/page numbers should be checked against the actual publications.
4. Consider adding a system diagram as Fig. 1 (the ASCII diagram in
   `PAPER.md` / the questionnaire is a good basis for a proper figure).
