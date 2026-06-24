# Research dossier — shanehsazzadeh2023unlocking

Phase 2 scientific review. Read-only: no converter/manifest/processed changes made.

## Verified facts
- **Paper:** Shanehsazzadeh A, et al. (AbSci). "Unlocking de novo antibody design with generative
  artificial intelligence." bioRxiv 2023.01.08.523187 (2024). *(paper)*
- **Antibody / antigen:** **trastuzumab** and AI-designed variants vs **human HER2/ERBB2**
  (UniProt **P04626**, extracellular domain aa23–645, domains I–IV; trastuzumab binds domain IV).
  *(paper + official database)*
- **Four tables:**
  - `kd_hher2_fab` (Fab format), `kd_hher2_mab` (mAb/IgG), `zerokd_trastuzumab`: raw `fitness` is the
    **pre-computed `-log10(K_D[M])`** column → `rank_label = fitness`, `neg_log10_kd_M`,
    higher_is_better, experimental. *(raw)*
  - `adcc_ec50`: raw `fitness` = **ADCC EC50 in pM**; converter `rank_label = -log10(EC50_pM · 1e-12)`
    → `neg_log10_ec50_M`, higher_is_better (lower EC50 = more potent), experimental. *(raw +
    deterministic transformation)*
- **Counts (verified via converter + validator):** adcc_ec50 13, kd_hher2_fab 13, kd_hher2_mab 13,
  zerokd_trastuzumab 422 — these match the README (an earlier `wc`-based count was off by one due to
  a missing trailing newline). *(raw/validator)*

## Current converter behavior
- kd tables: `_rl = float(raw)` (pre-logged). adcc: `_rl = -log10(raw · 1e-12)`.
- `antibody_type`: Fab for `kd_hher2_fab`, IgG for the others. Antigen hHER2 hard-coded (P04626 ECD).

## Conflicts between evidence and code
- **None on label/unit/direction/transform** — Kd and EC50 handling are correct.

## Unresolved issues
- **`adcc_ec50` assay_type = "binding" is imprecise.** ADCC EC50 is a **functional cytotoxicity
  potency** (effector-cell killing of HER2+ cells), not a binding affinity. `assay_type` would be
  more accurately "fitness"/"unknown"; left as-is per scope. The antigen context is still HER2.
  *(unresolved — metadata only; does not affect rank_label)*
- (Count correction) The Phase-2 draft listed off-by-one counts; the verified converter/validator
  output matches the README (adcc 13, kd_fab 13, kd_mab 13, zerokd 422). *(raw/validator)*

## Sources consulted
- bioRxiv 2023.01.08.523187 (AbSci de novo antibody design).
- UniProt P04626 (ERBB2_HUMAN) extracellular domain.
- Raw files `data/binding/shanehsazzadeh2023unlocking_*.csv`; `data/binding/README.md`.
