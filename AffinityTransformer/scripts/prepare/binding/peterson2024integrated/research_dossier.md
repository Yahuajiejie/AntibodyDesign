# Research dossier — peterson2024integrated

Phase 2 scientific review. Read-only: no converter/manifest/processed changes made.

## Verified facts
- **Paper:** Petersen BM, et al. "An integrated technology for quantitative wide mutational scanning
  of human antibody Fab libraries." *Nature Communications* 15:3974, 2024.
  DOI: 10.1038/s41467-024-48072-z (MAGMA-seq). *(paper)*
- **Antibody:** human **Fab** libraries (variable domains provided as heavy + light). *(paper)*
- **Antigen:** influenza **H1 hemagglutinin (HA)**; the study used a mixed pool against SARS-CoV-2 S1
  and influenza HA. The exact H1 HA strain/construct is not pinned down from the main text. *(paper)*
- **Assay:** MAGMA-seq — K_D extracted by fitting isogenic yeast-displayed Fab titrations to a
  binding isotherm. Reported **K_D in nM**. *(paper)*
- **Two tables:**
  - `ab_H1HA_kd`: raw `fitness` = apparent K_D in **nM** (range 2.68–2490.81 nM, median ≈ 25 nM —
    consistent with Fab-HA affinities). `rank_label = -log10(K_D[nM]·1e-9) = -log10(K_D[M])`;
    **higher_is_better**; experimental. *(paper + raw + deterministic transformation)*
  - `ab_H1HA_binary`: raw `fitness` ∈ **{0,1}** (bind / no-bind). `label_kind = binary`,
    `metric_name = binary`, **higher_is_better** (1 = bind). *(raw)*
- **Counts (raw):** binary = 1071 rows; kd = 1040 rows (both match README). *(raw)*

## Current converter behavior
- Both converters read `fitness`; `heavy`/`light` → chains (orientation correct: `EVQL…` VH,
  `EIVM…` VL).
- kd: `rank_label = -log10(fitness · 1e-9)` (column treated as nM) → `neg_log10_kd_M`,
  `-log10(KD/M)`, `higher_is_better`, experimental. **nM interpretation verified by the paper.**
- binary: `rank_label = float(fitness)` (0/1), `metric_name = binary`, `label_kind = binary`,
  `higher_is_better`, separate binary group.
- Both: `antigen_key = H1_HA`, `antigen_sequence = None`, `antigen_source = missing`,
  `antibody_type = IgG`.

## Conflicts between evidence and code
- **`antibody_type = IgG` is inaccurate (LOW–MEDIUM).** The paper builds and measures human **Fab**
  libraries. Recommend `antibody_type = Fab`. *(paper)*

## Unresolved issues
- **H1 HA antigen sequence:** not encoded (`antigen_source = missing`). The exact H1 HA
  strain/construct used as bait is not confirmed from the sources reviewed; leaving it `missing` is
  defensible, but it could be `retrieved` if the supplement specifies the strain. *(unresolved —
  acceptable as missing)*
- **binary vs continuous separation:** the binary table must stay a **separate group/label_kind**
  and never be pooled into continuous Spearman. *(repository convention — already honored)*

## Sources consulted
- Nature Communications DOI 10.1038/s41467-024-48072-z; PMC11087541; bioRxiv 2024.01.16.575852.
- Raw files `data/binding/peterson2024integrated_ab_H1HA_{binary,kd}.csv`; `data/binding/README.md`.
