# Research dossier — jain2024assessment

Phase 2 scientific review. Read-only: no converter/manifest/processed changes made.

## Verified facts
- **Paper:** Jain T, et al. (Adimab LLC + Sanofi). "Assessment and incorporation of in vitro
  correlates to pharmacokinetic outcomes in antibody developability workflows." *mAbs* 16(1),
  2024. DOI: 10.1080/19420862.2024.2384104 (open access; PMC11296533). *(paper)*
- **Antibodies:** germline-derived human antibody panels (heavy + light variable domains).
  README labels them "Fv … germline". *(paper + raw)*
- **Antigens:** the two lysozyme tables are control/anchor antigens measured on Octet BLI.
  - **Hen egg-white lysozyme (HEL)** — UniProt **P00698**, mature chain **aa19–147 (129 aa)**.
    Converter `ANTIGEN_SEQ` matches UniProt mature sequence **exactly**. *(official database — verified)*
  - **Mouse lysozyme C-2** — UniProt **P08905**, mature chain **aa19–148 (130 aa)**. Converter
    `ANTIGEN_SEQ` matches UniProt mature sequence **exactly**. *(official database — verified)*
- **Assay / metric:** Octet BLI dissociation constant in **molar (M)**.
  - Hen table: **monovalent** K_D (`Octet b-Hen Lysozyme Kd Monovalent (M)`).
  - Mouse table: **avidity** K_D on **IgG vs b-Mouse Lysozyme-Fc** (`Octet IgG KD … Avid (M)`).
  *(raw column headers + paper)*
- **Metric / direction:** raw `fitness` = `neg_log_kd` = `-log10(K_D[M])`; **higher_is_better**.
  Hen fitness range 6.30–9.52; mouse 6.40–7.00. *(deterministic transformation; raw)*
- **label_kind:** experimental. *(paper)*
- **Counts (raw):** Hen = 31 rows; mouse = 2 rows. No nulls, no duplicate (heavy,light). *(raw)*

## Current converter behavior
- Two converters read the **`fitness`** column directly as `rank_label` (already -log10 K_D).
- `heavy`/`light` → `heavy_chain`/`light_chain` (raw `heavy` = `EVQLV…/QLQLQ…` VH; raw `light` =
  `DIQMT…` VL — orientation correct).
- `antibody_type = IgG`, `antigen_source = retrieved`, `label_kind = experimental`,
  `metric_direction = higher_is_better` (set inline), one group per table.

## Conflicts between evidence and code
- **None affecting label/unit/direction/antigen identity** — antigen sequences verified against
  UniProt; units/direction correct.
- **Antibody-type granularity (minor):** `antibody_type = IgG` is the developability format, but
  the Hen K_D is a **monovalent** measurement and the raw provides only the Fv (VH+VL). Acceptable
  as a reasonable choice; the monovalent-vs-IgG distinction is captured by `assay_name`/table.

## Unresolved issues
- **Avidity vs monovalent comparability:** the mouse table is an **avidity** (bivalent IgG) K_D,
  not an intrinsic monovalent affinity. It is a **separate table/group**, so within-group ranking
  is valid; avidity and monovalent values must **not** be pooled. Documented, not a blocker.
  *(paper/raw)*
- **Mouse antigen construct:** experimental antigen is **b-Mouse-Lysozyme-Fc** (lysozyme fused to
  Fc, biotinylated). Converter uses the **lysozyme moiety only** (Fc fusion partner omitted) — a
  reasonable choice (Fc is an immobilization tag), but the exact construct is a modeling decision.
  *(reasonable implementation choice)*

## Sources consulted
- mAbs 2024 article (DOI 10.1080/19420862.2024.2384104); PubMed 39083118; PMC11296533.
- UniProt P00698 (LYSC_CHICK) and P08905 (LYZ2_MOUSE) FASTA — sequence verification.
- Raw files `data/binding/jain2024assessment_{Hen_Lys_kd,mouse_Ly_kd}.csv`.
- `data/binding/README.md`.
