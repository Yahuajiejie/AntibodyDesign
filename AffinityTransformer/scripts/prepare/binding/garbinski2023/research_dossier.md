# Research dossier — garbinski2023

Phase 2 scientific review. Read-only: no converter/manifest/processed changes made.

## Verified facts
- **Source:** No peer-reviewed publication located. `data/binding/README.md` records the source as
  **"Provided from GSK"** (2023), paper = None. A targeted search returned no Garbinski 2023
  antibody-affinity publication. Treat as an **internal/unpublished GSK dataset**. *(repository convention;
  absence of primary source confirmed)*
- **Antibody:** paired heavy + light variable domains (raw `heavy` = `VQLVESGGG…` VH; raw `light` =
  `DIQMT…` VL — orientation correct). *(raw)*
- **Metric / direction:** raw `KD` = dissociation constant in **M**; raw `-log (KD (M))` and
  `fitness` = `-log10(K_D[M])`. fitness range 7.52–10.9 (K_D ≈ 3e-8 to 1.3e-11 M). Smaller K_D =
  stronger ⇒ larger -log10 = better ⇒ **higher_is_better**. *(deterministic transformation; raw)*
- **label_kind:** experimental. *(raw — real K_D measurements)*
- **Count (raw):** **81 rows** (corrects an earlier "80" miscount; matches README). No nulls, no
  duplicate (heavy,light). *(raw)*

## Current converter behavior
- Reads `fitness` as `rank_label` (already -log10 K_D).
- `heavy`/`light` → `heavy_chain`/`light_chain` (orientation correct).
- `antibody_type = Fv`, `antigen_key = unknown_antigen`, `antigen_name = "Unknown proprietary
  antigen"`, `antigen_sequence = null`, `antigen_source = missing`, one group.

## Conflicts between evidence and code
- **None on label/unit/direction/chain orientation** — all consistent with the raw file.

## Unresolved issues
- **Antigen identity is undisclosed/proprietary (GSK).** Cannot be verified against any primary
  source. The converter's `unknown_antigen` / `antigen_source = missing` is the only defensible
  encoding. *(unresolved — by design, documented)*
- **Grouping assumption:** all 81 records are placed in **one group** under `unknown_antigen`,
  which assumes they share a single antigen/assay context. Plausible for a single GSK campaign but
  **not source-verifiable**. If the table actually mixes antigens, group ranking would be invalid.
  *(unresolved — reasonable convention, flagged)*

## Sources consulted
- `data/binding/README.md` (source attribution: GSK, no paper).
- Web search for "Garbinski 2023 antibody Kd GSK" — no matching primary publication found.
- Raw file `data/binding/garbinski2023_kd.csv`.
