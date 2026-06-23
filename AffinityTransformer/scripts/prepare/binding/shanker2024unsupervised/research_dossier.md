# Research dossier — shanker2024unsupervised

Phase 2 scientific review. Read-only: no converter/manifest/processed changes made.

## Verified facts
- **Paper:** Shanker VR, Bruun TUJ, Hie BL, Kim PS. "Unsupervised evolution of protein and antibody
  complexes with a structure-informed language model." *Science* 385:46–53, 2024.
  DOI: 10.1126/science.adk8946. Code/data: github.com/varun-shanker/structural-evolution. *(paper)*
- **Antibodies (two therapeutic anti-SARS-CoV-2 clinical antibodies, evolved with ESM-IF1):**
  - **Ly1404 = LY-CoV1404 (bebtelovimab).** *(paper)*
  - **SA58 = BD55-5840.** *(paper)*
  Each table is the parent ("WT") plus designed variants (heavy and/or light mutated); the raw
  `design`/`Design` column names the variant. *(raw + paper)*
- **Two assay types:**
  - **Neutralization IC50** (IC50 tables) — raw column `Avg Neutralization IC50 (ng/µL)` =
    `fitness`; **unit ng/µL**. *(raw)*
  - **Binding K_D by SPR** (Kd tables) — raw column `Kd avg (M)` = `fitness`; **unit M**. *(raw)*
- **Antigen:** SARS-CoV-2 spike of the named variant (Wuhan, BA.1, BQ.1.1, XBB.1.5). Wuhan RBD =
  UniProt **P0DTC2 aa319–541 (223 aa)** (verified length; same RBD used across this repo).
  *(official database)*
- **Direction / transform:** lower IC50 and lower K_D = better ⇒ `rank_label = -log10(value)` and
  **higher_is_better**. *(deterministic transformation)*
- **label_kind:** experimental. *(raw)*
- **Counts (raw):** Ly1404-BQ.1.1_IC50=50, Ly1404-BQ.1.1_Kd=36, Ly1404_Wuhan_IC50=**32**
  (README says 33), SA58-BA.1_IC50=19, SA58-BQ.1.1_IC50=49, SA58-BQ.1.1_Kd=7, SA58-XBB.1.5_Kd=30.
  *(raw)*

## Current converter behavior
- 7 converters; each reads `fitness`, `heavy`, `light`, and `design`/`Design` (→ `antibody_id`).
- IC50: `metric_name=neg_log10_ic50`, `metric_unit=-log10(IC50 ng/µL)`, `rank_label=-log10(IC50)`.
- Kd: `metric_name=neg_log10_kd_M`, `metric_unit=-log10(KD/M)`, `rank_label=-log10(Kd_M)`.
- All: `antibody_type=IgG`, `higher_is_better`, experimental, one group per table.
- Antigen: **only `Ly1404_Wuhan_IC50` has a retrieved sequence** (Wuhan RBD). All six variant tables
  have `antigen_sequence=None`, `antigen_source=missing`, and use names like `CoV2_BQ11_Spike`.

## Conflicts between evidence and code
- **None on label direction / unit basis / transform** — IC50 and Kd directions and `-log10`
  transforms are correct.

## Unresolved issues
1. **Non-physical Kd sentinels (HIGH, affects label meaning).** `SA58-XBB.1.5_Kd` contains a
   `Kd avg (M)` value of **7.64e+19** (the converter turns it into `rank_label ≈ -19.9`, ranked as
   the worst binder), and `SA58-BQ.1.1_Kd` contains a **0** (→ `-log10(0)=inf` → `rank_label=None`
   → dropped). These look like "no measurable binding" placeholders/errors handled **inconsistently**
   (one kept as an extreme value, one dropped). A single explicit policy (drop, or encode as a
   defined non-binder) is needed. *(unresolved)*
2. **Antigen identity incomplete/inconsistent (MEDIUM).** Only Wuhan is `retrieved`; the BA.1/BQ.1.1/
   XBB.1.5 RBD/spike sequences are derivable (defined mutation sets on the Wuhan RBD) but are marked
   `missing`. Also the SPR Kd tables (which bind **RBD**) carry antigen names like "…Spike". Variant
   antigen sequences should either be derived (as AbRank does for RBD mutants) or remain `missing`
   with that limitation documented. *(unresolved — kept missing is acceptable but incomplete)*
3. **IC50 unit ng/µL** is a relative potency, not molar; `-log10(ng/µL)` is monotonic so within-group
   ranking is valid, but values are not molar-comparable across studies. *(repository convention)*
4. **Replicates:** `SA58-XBB.1.5_Kd` has 4 duplicate `(heavy,light)` pairs. *(raw; annotation-stage)*
5. **README count** off by one for `Ly1404_Wuhan_IC50` (33 vs raw 32). *(raw)*

## Sources consulted
- Science DOI 10.1126/science.adk8946; PMC11616794; PubMed 38963838; github varun-shanker/structural-evolution.
- UniProt P0DTC2 (SARS-CoV-2 spike) RBD aa319-541.
- Raw files `data/binding/shanker2024unsupervised_*.csv`; `data/binding/README.md`.
