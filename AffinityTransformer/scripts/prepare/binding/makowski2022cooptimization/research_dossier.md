# Research dossier — makowski2022cooptimization

Phase 2 scientific review. Read-only: no converter/manifest/processed changes made.

## Verified facts
- **Paper:** Makowski EK, et al. (Tessier lab). "Co-optimization of therapeutic antibody affinity and
  specificity using machine learning models that generalize to novel mutational space." *Nature
  Communications* 13:3788, 2022. DOI: 10.1038/s41467-022-31457-3. *(paper)*
- **Antibody:** **emibetuzumab** (clinical anti-**c-Met/MET**) with CDR mutational libraries. The
  paper sorted libraries for high/low **affinity** (on-target) and **non-specific** (off-target)
  binding and deep-sequenced enriched libraries → enrichment-based binding signals (0–1). *(paper)*
- **Two antigens:**
  - **ANT = MET** (c-Met, UniProt **P08581**, SEMA extracellular region aa25–519) — the on-target
    antigen. *(paper + official database)*
  - **OVA = ovalbumin** (UniProt **P01012**) — the **off-target non-specificity reagent**. *(paper)*
- **Metric:** `rel_binding_signal` (relative binding signal 0–1); raw `fitness` used directly
  (`rank_label = fitness`), experimental. *(raw)*
- **Four tables (one record per row, verified via converter + validator):** `igg_ant` (96),
  `igg_ova` (96), `iso_ant` (126), `iso_ova` (126) — these match the README (an earlier `wc`-based
  count was off by one due to a missing trailing newline). `igg` and `iso` are **distinct CDR-variant
  panels** (different HCDR3 sequences). *(raw/validator)*

## Current converter behavior
- All four: `_rl = float(fitness)`, `metric_name=rel_binding_signal`, `metric_direction=
  higher_is_better`, `antibody_type=IgG`, antigen MET (ant tables) or Ovalbumin (ova tables),
  retrieved.

## Conflicts between evidence and code
- **None that are unambiguous errors** (target identities and metric are correct).

## Unresolved issues
1. **OVA direction semantics (MEDIUM).** The `*_ova` tables measure **off-target** binding.
   `metric_direction = higher_is_better` correctly means "higher signal ranks higher", but
   biologically **lower** OVA binding is the desirable (more specific) property. These rankings encode
   binding-signal magnitude, **not** antibody quality, and must not be interpreted as "better
   antibody" without inversion. The paper's goal is to *maximize* MET affinity while *minimizing* OVA
   non-specificity. *(unresolved — documented semantic caveat; not a hard error)*
2. **`iso` vs `igg` panel meaning (LOW).** Two distinct variant panels against the same antigens; the
   exact distinction (sort gate / sub-library) is not pinned down from the abstract. Each is its own
   table/group; does not affect label/unit/direction. *(unresolved)*
3. **Raw filename typo (LOW).** `iso_ant`'s raw file is `makowksi2022cooptimization_iso_ant.csv`
   (misspelled "makowksi"), and the converter `SOURCE_FILE` matches that misspelling, so it runs
   correctly. Inconsistent with the other three (`makowski…`). *(raw quirk; renaming is out of scope)*
4. (Count correction) The Phase-2 draft listed off-by-one counts; verified output matches the README
   (igg_ant/igg_ova 96, iso_ant/iso_ova 126). *(raw/validator)*

## Sources consulted
- Nature Communications DOI 10.1038/s41467-022-31457-3; PMC9249733.
- UniProt P08581 (MET_HUMAN) SEMA region; P01012 (OVAL_CHICK) ovalbumin.
- Raw files `data/binding/makow*si2022cooptimization_*.csv`; `data/binding/README.md`.
