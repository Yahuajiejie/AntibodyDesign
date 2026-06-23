# Research dossier — koenig2017mutational

Phase 2 scientific review. Read-only: no converter/manifest/processed changes made.

## Verified facts
- **Paper:** Koenig P, et al. "Mutational landscape of antibody variable domains reveals a switch
  modulating the interdomain conformational dynamics and antigen binding." *PNAS* 114(4):E486–E495,
  2017. DOI: 10.1073/pnas.1613231114. *(paper)*
- **Antibody:** **G6.31** — a high-affinity anti-VEGF-A antibody (variable domains; Fv/Fab format).
  *(paper + README)*
- **Antigen:** **VEGF-A** (human). Target identity correct. *(paper)*
- **Metric / unit / direction:** raw `fitness` = `-log( KD (M) )` (matches the `-log( KD (M) )`
  column). `rank_label = -log10(K_D[M])`; **higher_is_better**. fitness range 7.22–12.16 ⇒
  K_D ≈ 6e-8 to 7e-13 M (sub-nM to pM), consistent with high-affinity anti-VEGF binders.
  *(raw + deterministic transformation)*
- **label_kind:** experimental. *(raw)*
- **Count (raw):** ~**4275 data rows** (README says 4276; off by one). *(raw)*

## Current converter behavior
- Reads `fitness` directly as `rank_label`; `heavy`/`light` → chains (orientation correct:
  `EAQL…/ECQL…` VH variants, `DIQM…` VL).
- `metric_name = neg_log10_kd_M`, `metric_unit = -log10(KD/M)`, `higher_is_better`,
  `label_kind = experimental`, `assay_name = SPR`, `antibody_type = Fv`.
- `antigen_key = VEGF_A`, `antigen_sequence = <reconstructed VEGF165>`, `antigen_source = retrieved`.

## Conflicts between evidence and code
- **None on target / label / unit / direction** — VEGF-A target, `-log10(K_D[M])`, higher-is-better
  all correct.

## Unresolved issues
- **VEGF construct sequence (MEDIUM — needs verification).** The converter's `ANTIGEN_SEQ` is a
  **non-standard reconstruction** ("VEGF165 reconstructed from long-isoform coordinates aa207-320 +
  aa346-395 with VEGF165 inserted"). The exact VEGF construct used in the binding assay is not
  confirmed; G6-class antibodies are typically characterized against the VEGF-A **receptor-binding
  domain (VEGF8-109)** rather than full VEGF165. The sequence should be re-verified against the
  paper/supplement and UniProt P15692 before being treated as authoritative. Target (VEGF-A) is
  correct; the exact residues are **unresolved**. *(unresolved)*
- `assay_name = "SPR"` is plausible but not separately verified here. *(low)*

## Sources consulted
- PNAS DOI 10.1073/pnas.1613231114 (G6.31, anti-VEGF-A, mutational landscape).
- UniProt P15692 (VEGFA_HUMAN) — referenced for the construct (sequence not finalized here).
- Raw file `data/binding/koenig2017mutational_kd_g6.csv`; `data/binding/README.md`.
