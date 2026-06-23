# Research dossier — kirby2024retrospective

Phase 2 scientific review. Read-only: no converter/manifest/processed changes made.

## Verified facts
- **Paper:** Kirby MB, Petersen BM, Faris JG, Kells SP, Sprenger KG, Whitehead TA. "Retrospective
  SARS-CoV-2 human antibody development trajectories are largely sparse and permissive." *PNAS*
  2025. DOI: 10.1073/pnas.2412787122 (PMC11789010). Whitehead lab, U Colorado Boulder — the same
  lab/method family as peterson2024integrated (yeast-display deep mutational scanning). *(paper)*
- **Antibodies:** human anti-SARS-CoV-2 Spike antibodies and their development-trajectory variants.
  The paper studies **eight** antibodies; the repo README lists CC12.1 and 1-20 as exemplars. The
  raw tables provide only `heavy`/`light` sequences (no antibody-name column), so multiple lineages
  are intermixed by sequence. *(paper + raw)*
- **Antigen:** SARS-CoV-2 **Wuhan-Hu-1 RBD** = UniProt **P0DTC2 aa319–541 (223 aa)** (verified
  length; same RBD used across this repo). *(official database)*
- **Two tables:**
  - `ab-SARSCoV2_kd`: column `Kd [M]` (= label). Values range **3.3–2950 (median 36)** — physically
    impossible in molar; consistent with **nM** and with the Whitehead lab's yeast-display K_D,App
    convention (nM, as in peterson2024). `rank_label = -log10(K_D[nM]·1e-9) = -log10(K_D[M])`,
    **higher_is_better**, experimental. *(raw + paper/lab convention)*
  - `ab-SARSCoV2_binary_kd`: column `KD [bind/no bind]` ∈ **{0,1}**. `label_kind = binary`,
    `metric_name = binary`, **higher_is_better** (1 = bind). *(raw)*
- **Counts (raw):** kd = 869 rows; binary = 1407 rows (both match README). *(raw)*

## Current converter behavior
- Both read `heavy`/`light`; kd reads `Kd [M]`, binary reads `KD [bind/no bind]`.
- kd: `rank_label = -log10(Kd_nM · 1e-9)` (header column **treated as nM**), `neg_log10_kd_M`,
  `-log10(KD/M)`, higher_is_better, experimental. The converter header comment already states
  "Values treated as nM. If confirmed otherwise, update."
- binary: `rank_label = float(value)` (0/1), `metric_name=binary`, `label_kind=binary`.
- Both: `antigen_key=CoV2_Wuhan_RBD`, `antigen_sequence=P0DTC2 aa319-541`, `antigen_source=retrieved`,
  `antibody_type=IgG`, one group per table.

## Conflicts between evidence and code
- **`Kd [M]` header is mislabeled (the values are nM).** The converter already corrects for this
  (treats as nM); the **raw column header** is the thing that is wrong, not the converter. *(raw vs
  converter — converter behavior is correct)*

## Unresolved issues
1. **Antibody heterogeneity / grouping (MEDIUM).** Up to eight distinct antibody lineages are pooled
   into one group per table (keyed only on the shared Wuhan RBD antigen). Affinity ranking against a
   single antigen is defensible, but the lineages may target different RBD epitopes; cross-lineage
   ranking should be interpreted with care. No antibody-name column exists to separate them.
   *(unresolved — grouping interpretation)*
2. **Heavy replication (annotation-stage).** kd has **253** duplicate `(heavy,light)` pairs (of 869);
   binary has **368** (of 1407). Duplicate sequences may carry different labels (true replicates /
   trajectory re-measurements); base-records convention keeps one record per raw row, deferring
   replicate aggregation to the annotation stage. *(raw)*
3. **K_D unit basis.** nM is strongly supported (data range + lab convention) but the literal header
   says `[M]`; ideally confirm against the PNAS methods/supplement. The molar reading is physically
   impossible, so the nM interpretation is adopted. *(paper/lab convention; residual confirmation)*

## Sources consulted
- PNAS DOI 10.1073/pnas.2412787122; PubMed 39841142; PMC11789010.
- UniProt P0DTC2 (SARS-CoV-2 spike) RBD aa319-541.
- Raw files `data/binding/kirby2024retrospective_ab-SARSCoV2_{kd,binary_kd}.csv`; `data/binding/README.md`.
