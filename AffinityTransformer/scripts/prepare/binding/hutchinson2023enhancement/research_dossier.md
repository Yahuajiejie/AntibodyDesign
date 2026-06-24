# Research dossier — hutchinson2023enhancement

Phase 2 scientific review. Read-only: no converter/manifest/processed changes made.

## Verified facts
- **Paper:** Hutchinson M, et al. (AstraZeneca + Johns Hopkins). "Toward enhancement of antibody
  thermostability and affinity by computational design in the absence of antigen." *mAbs* 16(1),
  2024. DOI: 10.1080/19420862.2024.2362775 (PMC11195458; bioRxiv 2023.12.19.572421). *(paper)*
- **Antibody:** parental **D44.1** (mouse monoclonal **anti-hen-egg-lysozyme**) plus ~200 DeepAb +
  single-point-DMS computationally designed variants. The raw `heavy`/`light` are the D44.1-variant
  variable domains; the WT D44.1 sequence is row 1 of every table (heavy matches the canonical D44.1
  used in warszawski2019). *(paper + raw cross-check)*
- **Antigen:** **hen egg-white lysozyme (HEL)** = UniProt **P00698**, mature chain **aa19–147
  (129 aa)** — identical to the verified HEL sequence used in jain2024assessment and the corrected
  warszawski2019. *(official database)*
- **Two assay formats (separate tables):** **Fab** binding kinetics (monovalent affinity) and **IgG**
  binding kinetics (**single-point apparent affinity**, avidity-influenced; IgG expressed on the
  NIP228 framework). *(paper + raw column headers)*
- **Metric / unit / direction:** raw `fitness` = the **pre-computed `-log10(K_D[M])`** column
  (e.g. K_D=2.55e-8 M → fitness=7.5935). `rank_label = fitness` (used directly), `neg_log10_kd_M`,
  `-log10(KD/M)`, **higher_is_better**, experimental. *(raw + deterministic transformation)*
- **Counts (verified via pandas):** multikd_fab 15, multikd_igg 15, singlekd_fab 22, singlekd_igg 23,
  top200kd_fab 50, top200kd_igg 182, top27kd_fab 28, top27kd_igg 27 — these match the README. *(raw)*

## Current converter behavior
- 8 converters; each reads `fitness` directly (`_rl = float(raw)`); antigen HEL hard-coded retrieved;
  `antibody_type = Fab` (fab tables) / `IgG` (igg tables); one group per table.

## Conflicts between evidence and code
- **None.** Antibody (D44.1 anti-HEL), antigen (HEL P00698), Fab/IgG formats, the M-based pre-computed
  `-log10(K_D[M])` label (used without re-transforming), direction, and `label_kind` are all correct.

## Unresolved issues
- **README unit label is misleading (not a converter error).** The README column reads
  "-log (Kd [nM])", but the actual `fitness` and converter output are **-log10(K_D in M)** (M-based,
  verified from the raw `-log10(KD (M))` column). The converter is correct; the README text is wrong
  (out of scope to change). *(raw vs README)*
- **Fab vs IgG comparability.** IgG values are **single-point apparent** affinities (avidity-
  influenced), not monovalent Fab K_D. They are kept in **separate tables/groups** and must not be
  pooled. *(paper)*
- **Cross-table redundancy / replicates (annotation-stage).** The tables are overlapping design
  subsets (WT D44.1 appears in all 8; top27 ⊂ top200), and within-table duplicate `(heavy,light)`
  pairs exist (top200kd_fab 2, top27kd_fab 1, top27kd_igg 1). One record per raw row is kept;
  measurement-family aggregation is deferred to the annotation stage. *(raw)*
- **NIP228** is the IgG expression framework / control mAb, **not** an antigen; the antigen is HEL.
  *(paper)*

## Sources consulted
- mAbs 2024 DOI 10.1080/19420862.2024.2362775; PMC11195458; PubMed 38899735; bioRxiv 2023.12.19.572421.
- UniProt P00698 (LYSC_CHICK) mature sequence; warszawski2019 D44.1 cross-check.
- Raw files `data/binding/hutchinson2023enhancement_*.csv`; `data/binding/README.md`.
