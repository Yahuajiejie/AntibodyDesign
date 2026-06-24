# Research dossier — shanehsazzadeh2024igdesign

Phase 2 scientific review. Read-only: no converter/manifest/processed changes made.

## Verified facts
- **Paper:** Shanehsazzadeh A, et al. (AbSci). "IgDesign: In vitro validated antibody design against
  multiple therapeutic antigens using inverse folding." bioRxiv 2023.12.08.570889 (2024).
  Code: github.com/AbSciBio/igdesign. *(paper)*
- **Method:** inverse-folding design of antibody HCDR3 / HCDR123 against therapeutic antigen–antibody
  complexes; binders validated by **SPR**, K_D reported in **nM**. Each table holds the reference
  therapeutic ("Positive Control") plus designed variants (raw `Method`/`HCDRs Designed` columns).
  *(paper + raw)*
- **Antibody→target pairings (clinical facts):** Afasevikumab→IL-17A, Bimagrumab→ACVR2B,
  Eculizumab→C5, Osocimab→FXI, Spesolimab→IL-36R, Tezepelumab→TSLP, Utomilumab→TNFRSF9 (4-1BB).
  `antibody_type = IgG`. *(paper + clinical knowledge)*
- **Antigens (retrieved, hard-coded with UniProt notes):**
  - IL17A = Q16552 mature **aa24–155** — **verified exact** against UniProt FASTA.
  - ACVR2B = Q13705 ECD aa19–137; C5 = P01031 full-length aa1–1676; FXI = P03951 mature aa19–625;
    IL36R = Q9HB29 ECD aa20–335 (note records a prior wrong accession Q9Y5U5/GITR that was corrected);
    TSLP = Q969D9 mature aa29–159; TNFRSF9 = Q07011 CRD2–CRD4 aa24–186. Targets verified; exact
    construct boundaries per the UniProt notes (spot-verification recommended). *(official database)*
- **Metric / direction / transform:** raw `fitness` = K_D in **nM**; converter `rank_label =
  -log10(K_D_nM · 1e-9)` → `neg_log10_kd_M`, `-log10(KD/M)`, **higher_is_better**, experimental.
  *(raw + deterministic transformation)*
- **Counts (raw):** Afasevikumab 13, Bimagrumab 24, Eculizumab 34, Osocimab 47, Spesolimab 40,
  Tezepelumab 127, Utomilumab 36 (match README). *(raw)*

## Current converter behavior
- 7 converters; each reads `fitness`, `heavy`, `light`; `_rl = -log10(fitness · 1e-9)`. Antigen
  sequence/source/note hard-coded; one group per (antibody, antigen) table.

## Conflicts between evidence and code
- **None on label/unit/direction.** The nM→M conversion is correct; the README label string
  "-log(Kd[nM])" describes the raw column, while the output is correctly M-based.

## Unresolved issues
- **Antigen construct boundaries** are hard-coded (only IL17A verified exact here); the others are
  plausible per their UniProt notes but the exact assay constructs were not re-verified one-by-one.
  *(spot-verify recommended; targets are verified)*
- **Unit convention** is M-based here vs nM-based in rosace2023automated (same batch) — each is
  internally consistent; not cross-comparable. *(repository convention)*

## Sources consulted
- bioRxiv 2023.12.08.570889; github AbSciBio/igdesign; AbSci IgDesign whitepaper.
- UniProt Q16552 (IL17A) FASTA — verified; Q13705/P01031/P03951/Q9HB29/Q969D9/Q07011 (notes).
- Raw files `data/binding/shanehsazzadeh2024igdesign_*.csv`; `data/binding/README.md`.
