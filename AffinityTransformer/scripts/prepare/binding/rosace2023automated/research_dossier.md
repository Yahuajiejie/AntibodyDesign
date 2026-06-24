# Research dossier — rosace2023automated

Phase 2 scientific review. Read-only: no converter/manifest/processed changes made.

## Verified facts
- **Paper:** Rosace A, et al. "Automated optimisation of solubility and conformational stability of
  antibodies and proteins." *Nature Communications* 14, 2023. DOI: 10.1038/s41467-023-37668-6.
  *(paper)*
- **Antibodies / antigen:** **adalimumab** and **golimumab** (both clinical anti-**TNF-α** antibodies)
  and engineered variants; antigen = **TNF-α** (`antigen_key=TNFa`, retrieved). *(paper + clinical
  knowledge)*
- **Metric / unit / direction:** raw `fitness` = the pre-computed **`-log(Kd [nM])`** column (last
  column; a `-log10(KD/M)` column is also present for reference). `rank_label = fitness`,
  `neg_log10_kd_nM`, `-log10(KD/nM)`, **higher_is_better**, experimental. Row 0 (adalimumab WT):
  K_D=1.3e-10 M = 0.13 nM → fitness = -log10(0.13) = 0.886 (matches raw). *(raw + deterministic
  transformation)*
- **Counts (raw):** kd_adalimumab 14, kd_golimumab 5. *(raw)*

## Current converter behavior
- Both read `fitness` directly (`_rl = float(raw)`); `antibody_type = Fv`; antigen TNFa hard-coded
  retrieved; one group per table.

## Conflicts between evidence and code
- **None on label/unit/direction.**

## Unresolved issues
- **Unit convention (nM-based)** differs from igdesign/unlocking (M-based) in the same batch. Each is
  internally consistent; values are offset by a constant +9, so within-group ranking is identical and
  cross-study comparison is not done. *(repository convention)*
- The paper's primary subject is solubility/stability optimisation; these tables are the **affinity**
  (anti-TNF K_D) measurements for the same engineered variants. *(paper)*
- README lists the year as 2022; the DOI/journal record is 2023. *(minor metadata)*

## Sources consulted
- Nature Communications DOI 10.1038/s41467-023-37668-6.
- Raw files `data/binding/rosace2023automated_kd_{adalimumab,golimumab}.csv`; `data/binding/README.md`.
