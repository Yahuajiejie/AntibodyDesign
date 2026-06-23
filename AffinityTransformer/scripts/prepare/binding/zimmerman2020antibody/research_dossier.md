# Research dossier — zimmerman2020antibody

Phase 2 scientific review. Read-only: no converter/manifest/processed changes made.
**This study has a verified antibody-identity defect and a provenance mismatch — see Conflicts.**

## Verified facts
- **Antibody:** **4-4-20 anti-fluorescein** antibody and affinity-maturation variants. The cited
  PNAS paper states 4-4-20 acquired "two VL mutations (CDR1 + adjacent β-strand) and 10 VH
  mutations." Mature 4-4-20 binds fluorescein at K_D ≈ 1.2 nM; the raw table holds weaker
  intermediate variants (K_D ≈ 150 µM–~30 nM). *(paper)*
- **Antigen:** **fluorescein** — small-molecule hapten, **no protein sequence** ⇒
  `antigen_source = missing` is correct. *(paper)*
- **Metric / direction:** raw `KD (M)` = dissociation constant in M; raw `-log(KD M)` and
  `fitness` = `-log10(K_D[M])`. fitness range 3.82–7.52 (K_D 150 µM–~30 nM). **higher_is_better**.
  *(deterministic transformation; raw)*
- **label_kind:** experimental. *(raw)*
- **Count (raw):** **21 rows** (corrects an earlier "20" miscount); **1 duplicate (heavy,light)**
  pair (one repeated variant). No nulls. *(raw)*
- **Chain biology (basis for the swap finding):** `EVKL…` = heavy variable (VH); `DVVMTQ…` =
  kappa light variable (VL). Confirmed by cross-checking the **same 4-4-20 antibody in
  adams2017measuring**, where the raw is correctly ordered (heavy=EVKL, light=DVVMTQ). *(raw + cross-study)*

## Current converter behavior
- Reads `fitness` directly as `rank_label` (already -log10 K_D) — value correct.
- Maps raw column **`heavy` → `heavy_chain`** and raw column **`light` → `light_chain`** by name.
- `antibody_type = scFv`, `assay_name = "SPR"`, `antigen_source = missing`,
  `label_kind = experimental`, `metric_direction = higher_is_better`, one group.

## Conflicts between evidence and code
1. **HEAVY/LIGHT SWAPPED (HIGH severity — antibody identity).** In the raw file, the column named
   `heavy` actually contains the **VL** (`DVVMTQ…`) and the column named `light` contains the
   **VH** (`EVKL…`). The converter assigns by column name, so the output `heavy_chain` field holds
   the **light** chain and vice-versa. **Fix at implementation:** `heavy_chain ← raw 'light'`,
   `light_chain ← raw 'heavy'`. *(verified: biology + adams cross-check)*
2. **Provenance mismatch (MEDIUM).** `study_id = zimmerman2020antibody` but the only recorded DOI
   (`10.1073/pnas.0603282103`, README) is the **2006** PNAS paper "Antibody evolution constrains
   conformational heterogeneity by tailoring protein dynamics." The "2020" year is unexplained;
   the exact paper/table that this 21-variant K_D list was taken from is **not confirmed**.
   *(unresolved provenance)*
3. **`assay_name = "SPR"` unverified (LOW).** 4-4-20/fluorescein affinity is classically measured by
   **fluorescence-quenching titration**, not SPR. The method behind these K_D values is not
   confirmed. `assay_type = binding` is fine; `assay_name` is metadata only. *(unresolved metadata)*

## Unresolved issues
- The originating table/paper for the 21 variants and the exact assay method (items 2–3 above).
- `antibody_type`: labeled `scFv` (consistent with the 4-4-20 scFv used in adams); the raw provides
  separate VH/VL, so an `Fv` label would also be defensible. *(reasonable choice)*

## Sources consulted
- PNAS DOI 10.1073/pnas.0603282103 (paper identity + 4-4-20 VH/VL mutation description).
- adams2017measuring raw (canonical 4-4-20 VH/VL orientation cross-check).
- Raw file `data/binding/zimmerman2020antibody_4420_kd.csv`; `data/binding/README.md`.
