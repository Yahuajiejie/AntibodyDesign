# Research dossier — adams2017measuring

Phase 2 scientific review. Read-only: no converter/manifest/processed changes made.

## Verified facts
- **Paper:** Adams RM, Mora T, Walczak AM, Kinney JB. "Measuring the sequence-affinity
  landscape of antibodies with massively parallel titration curves." *eLife* 5:e23156.
  DOI: 10.7554/eLife.23156 (published 2016-12-30; commonly cited as 2017). *(paper)*
- **Antibody:** 4-4-20 anti-fluorescein **scFv** (PDB 1FLR); CDR1H and CDR3H variant
  libraries. The raw `format` column = `scFv` for every row, confirming the construct. *(paper + raw)*
- **Antigen:** **fluorescein** — a small-molecule hapten with **no protein sequence**.
  `antigen_source = missing` is correct. *(paper)*
- **Assay / method:** Tite-Seq (titeseq table) and flow-cytometry titration (flow table);
  both report **absolute dissociation constant K_D in molar (M)**. *(paper + raw column headers
  "Kd (Tite-Seq) [M]", "Kd (flow) [M]")*
- **Metric / direction:** raw `fitness` column = K_D in M (matches the `Kd [M]` column).
  `rank_label = -log10(K_D[M])`; smaller K_D = stronger binding ⇒ larger -log10 = better ⇒
  **higher_is_better**. *(deterministic transformation; direction is biology-verified)*
- **label_kind:** experimental. *(paper)*
- **Counts (raw):** flow = 15 rows; titeseq = 11,052 rows. No null heavy/light. *(raw)*

## Current converter behavior
- Two converters (`4420-fluorescein_kd-flow`, `4420-fluorescein_kd-titeseq`).
- Both read the **`fitness`** column and compute `rank_label = -log10(fitness)`.
- `heavy`/`light` raw columns map directly to `heavy_chain`/`light_chain` (orientation correct:
  raw `heavy` = `EVKL…` VH, raw `light` = `DVVMTQ…` VL — verified by cross-check with the
  canonical 4-4-20 VH/VL).
- `antibody_type = scFv`, `antigen_source = missing`, `label_kind = experimental`,
  `metric_direction = higher_is_better`, one group per table.
- Drop rule: drop if heavy chain missing/invalid or rank_label non-finite.

## Conflicts between evidence and code
- **None affecting correctness.** Chain orientation, units, direction, label_kind, antigen all
  match evidence.
- **Cosmetic only:** both converters define `KD_COL = "Kd (flow) [M]"` which is **dead/unused**
  code (the titeseq file's column is `Kd (Tite-Seq) [M]`); the converters read `fitness`, so
  output is unaffected. Recommend deleting the unused constant during implementation.

## Unresolved issues
- **titeseq duplicates:** 11,052 rows contain **8,245 duplicate `(heavy,light)` pairs**
  (~2,807 unique sequences). These are repeated/again-measured variants (CDR1H + CDR3H libraries
  share the WT background and bins). Base-records convention = one record per raw row with a
  traceable `source_row`; whether these should later be treated as replicates/aggregated is a
  **measurement-family/annotation-stage** question, **not** a base-records blocker. Flagged for
  the later entity-annotation phase. *(unresolved — out of base-records scope)*

## Sources consulted
- eLife article 23156 (DOI 10.7554/eLife.23156); PubMed 28035901; PMC5268739.
- Raw files `data/binding/adams2017measuring_4420-fluorescein_kd-{flow,titeseq}.csv`.
- `data/binding/README.md` (repository convention, DOI + direction).
