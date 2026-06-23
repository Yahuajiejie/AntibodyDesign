# Research dossier — warszawski2019

Phase 2 scientific review. Read-only: no converter/manifest/processed changes made.
**This study has a verified WRONG-ANTIGEN defect — see Conflicts.**

## Verified facts
- **Paper:** Warszawski S, et al. "Optimizing antibody affinity and stability by the automated
  design of the variable light-heavy chain interfaces." *PLoS Comput Biol* 15(8):e1007207, 2019.
  DOI: 10.1371/journal.pcbi.1007207. *(paper)*
- **Antibody:** **D44.1** — a mouse monoclonal **anti-hen-egg-white-lysozyme** antibody; the study
  applied yeast-display deep mutational scanning to D44.1. *(paper)*
- **Antigen:** **hen egg-white lysozyme (HEL)**, NOT VEGF. Confirmed by the canonical D44.1–HEL
  complex **PDB 1MLC** ("Monoclonal antibody Fab D44.1 raised against chicken egg-white lysozyme
  complexed with lysozyme"; WT K_D ≈ 137 nM, consistent with the raw data range). HEL =
  **UniProt P00698**, mature chain **aa19–147 (129 aa)** — the same sequence verified for
  jain2024assessment. *(paper + official database/structure)*
- **Assay:** yeast-display titration / SPR-class binding K_D. *(paper)*
- **Metric / unit / direction:** raw `fitness` = `-log(Kd [nM])` (the two columns are bit-identical
  across all rows). So `rank_label = -log10(K_D[nM])`; smaller K_D = stronger ⇒ larger value =
  better ⇒ **higher_is_better**. Note the unit is **nM-based**, differing from the M-based tables in
  this batch by a constant +9 offset (ranking is unaffected). *(raw + deterministic transformation)*
- **label_kind:** experimental. *(raw)*
- **Count (raw):** **2048 data rows** (README says 2049; off by one). *(raw)*

## Current converter behavior
- Reads `fitness` directly as `rank_label`; `heavy`/`light` → chains (orientation correct:
  `QVQL…` VH, `DIEL…` VL).
- `metric_name = neg_log10_kd_nM`, `metric_unit = -log10(KD/nM)`, `higher_is_better`,
  `label_kind = experimental`, `assay_name = SPR`.
- **`antigen_key = VEGF_A`, `antigen_name = "Vascular endothelial growth factor A (VEGF-A)"`,
  `antigen_sequence = <VEGF sequence>`, `ANTIGEN_SOURCE_NOTE = VEGF165 …`** — and `GROUP_ID`
  embeds `VEGF_A`.

## Conflicts between evidence and code
1. **WRONG ANTIGEN (HIGH — antigen identity).** The converter labels the antigen as **VEGF-A** with
   the VEGF165 sequence and note — **byte-identical to koenig2017mutational's antigen block** (a
   copy-paste error). D44.1's antigen is **hen egg-white lysozyme**, verified by the paper and PDB
   1MLC. **Fix at implementation:** set `antigen_key = HEL`, `antigen_name = "Hen egg lysozyme"`,
   `antigen_sequence = P00698 mature aa19-147 (129 aa)`, update the source note, and regenerate the
   `group_id` (which currently contains `VEGF_A`). *(verified)*

## Unresolved issues
- `assay_name = "SPR"`: the affinities come from yeast-display titration / DMS; SPR labeling is
  metadata only and unverified. *(low; unresolved metadata)*
- `antibody_type = Fv`: D44.1 is a Fab; the raw provides the variable domains, so `Fv` is
  acceptable. *(reasonable choice)*

## Sources consulted
- PLoS Comput Biol DOI 10.1371/journal.pcbi.1007207 (D44.1 = anti-lysozyme).
- RCSB **PDB 1MLC** / 1MLB (D44.1 Fab raised against chicken egg-white lysozyme); D44.1–HEL kinetics
  (K_D ≈ 137 nM).
- UniProt **P00698** (LYSC_CHICK) mature sequence.
- Raw file `data/binding/warszawski2019_d44_Kd.csv`; `data/binding/README.md`.
