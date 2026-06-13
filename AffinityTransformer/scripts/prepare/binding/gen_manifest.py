#!/usr/bin/env python3
"""Regenerate manifest.csv and antigen_missing_summary.csv from convert.py files.

Run from repo root:
    python3 scripts/prepare/binding/gen_manifest.py

Writes:
    scripts/prepare/binding/manifest.csv
    scripts/prepare/binding/antigen_missing_summary.csv
"""
import csv, re, sys
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[3]   # repo root
OUTDIR = Path(__file__).resolve().parent        # scripts/prepare/binding/
SCRIPTS_BASE = OUTDIR

# ─── helpers ──────────────────────────────────────────────────────────────────

def _extract(text, key, default=""):
    """Extract single-quoted or double-quoted string value of a constant."""
    m = re.search(
        rf'^{re.escape(key)}\s*=\s*["\']([^"\']*)["\']',
        text, re.MULTILINE
    )
    return m.group(1) if m else default


def _extract_multi(text, key, default=""):
    """For ANTIGEN_KEY that may appear in multi-antigen scripts as dict/logic."""
    m = re.search(
        rf'^{re.escape(key)}\s*=\s*["\']([^"\']*)["\']',
        text, re.MULTILINE
    )
    return m.group(1) if m else default


def _has_zip(source_file):
    return source_file.lower().endswith(".zip")


def _infer_notes(study_id, table_id, text, antigen_source, antigen_source_note, label_kind, source_file):
    notes = []
    if _has_zip(source_file):
        notes.append("zip; streaming parquet write")
    if "VHH" in text and "_strip_vhh" in text:
        notes.append("VHH signal peptide + His-tag stripped")
    if label_kind == "predicted":
        notes.append("predicted label (ML score)")
    if label_kind == "binary":
        notes.append("binary label (0/1)")
    if antigen_source == "missing":
        notes.append("antigen seq missing")
    if antigen_source == "retrieved" and antigen_source_note:
        notes.append(antigen_source_note)
    if "censored_measurement" in text:
        notes.append("censored values dropped")
    if "SKIP_ROWS" in text:
        m = re.search(r'SKIP_ROWS\s*=\s*(\d+)', text)
        if m and int(m.group(1)) > 0:
            notes.append(f"skip {m.group(1)} header rows")
    return "; ".join(notes) if notes else ""


# ─── blocked datasets (not in scripts, manually specified) ────────────────────

BLOCKED = [
    dict(
        study_id="rawat2022abcov", table_id="abcov",
        csv_name="rawat2022abcov_AbCoV.csv",
        antibody_type="IgG",
        antigen_key="missing",
        antigen_name="SARS-CoV-2 (no antigen column in CSV)",
        antigen_source="missing",
        metric_name="unknown",
        label_kind="unknown",
        status="blocked",
        notes="CSV has no antigen column; cannot construct group_id; need antigen info from paper",
    ),
    dict(
        study_id="rawat2022abcov", table_id="abcov2",
        csv_name="rawat2022abcov_AbCoV2.csv",
        antibody_type="IgG",
        antigen_key="missing",
        antigen_name="SARS-CoV-2 (no antigen column in CSV)",
        antigen_source="missing",
        metric_name="unknown",
        label_kind="unknown",
        status="blocked",
        notes="CSV has no antigen column; cannot construct group_id; need antigen info from paper",
    ),
]

# ─── antigen reference data for missing_summary ───────────────────────────────
# Manually curated: UniProt/PDB suggestions for antigens not in source CSVs.

ANTIGEN_REFS = [
    # phillips2021binding
    ("H1_HA_Solomon",        "Influenza A H1 HA (A/Solomon Islands/3/2006)", True,  "A/Solomon Islands/3/2006 HA",    "Influenza A",     "UniProt Q3HM16 or NCBI ABD59308; use ecto-domain aa 18-566"),
    ("H9_HA_HK",             "Influenza A H9 HA (A/HK/1073/99)",             True,  "A/HK/1073/99 HA",                "Influenza A",     "GenBank AF156378; use ecto-domain"),
    ("H1_HA_PR8",            "Influenza A H1 HA (PR8 strain)",               True,  "A/Puerto Rico/8/1934 HA",        "Influenza A",     "UniProt P03452; signal-peptide cleaved; mature aa 18-566"),
    ("H3_HA_HK68",           "Influenza A H3 HA (A/HK/1/68)",                True,  "A/Hong Kong/1/1968 HA",          "Influenza A",     "UniProt P03437; mature ecto-domain"),
    # hie2023efficient
    ("CoV2_WT_S6P",          "SARS-CoV-2 WT Spike S6P (hexapro)",            True,  "7LYL",                           "SARS-CoV-2",      "PDB 7LYL; 6-proline stabilised prefusion; use sequence from PDB SEQRES"),
    ("CoV2_Beta_S6P",        "SARS-CoV-2 Beta (B.1.351) Spike S6P",          True,  "7LYL+Beta mutations",            "SARS-CoV-2",      "Apply Beta mutations (K417N/E484K/N501Y) to 7LYL sequence"),
    ("CoV2_Omicron_S6P",     "SARS-CoV-2 Omicron (BA.1) Spike S6P",         True,  "7LYL+Omicron mutations",         "SARS-CoV-2",      "Apply BA.1 mutations to 7LYL sequence; 37 substitutions"),
    ("H1_HA_Solomon_MEDI",   "Influenza A H1 HA (MEDI context)",             True,  "A/Solomon Islands/3/2006 HA",    "Influenza A",     "Same as H1_HA_Solomon; see hie2023efficient supplementary"),
    ("H4_HA_Hubei",          "Influenza A H4 HA (A/Hubei/1/2010)",           True,  "A/Hubei/1/2010 HA",              "Influenza A",     "GenBank JF730435 or similar H4 strain; use ecto-domain"),
    ("H7_HA_HK16",           "Influenza A H7 HA (A/HK/2014/2016)",           True,  "A/Hong Kong/2014/2016 HA",       "Influenza A",     "GISAID or NCBI; use ecto-domain"),
    ("Ebola_GP",             "Ebola virus glycoprotein (mAb114 target)",     True,  "6MDT",                           "Zaire ebolavirus", "PDB 6MDT; use GP1+GP2 ecto-domain; remove mucin-like domain aa 313-461 if needed"),
    # rosace2023automated
    ("IL12B_golimumab",      "IL-12 p40 subunit (golimumab target)",         True,  "P29460",                         "Homo sapiens",    "UniProt P29460; mature form aa 23-328; golimumab also binds IL-23"),
    # koenig2017mutational
    ("VEGF_g6",              "VEGF (G6 antibody context)",                   True,  "P15692",                         "Homo sapiens",    "UniProt P15692; VEGF-A165; aa 27-232 mature form"),
    # jain2024assessment
    ("mLy_jain",             "Mouse lysozyme",                               True,  "P00695 or P08905",               "Mus musculus",    "UniProt P00695 (Lyz1) or P08905 (Lyz2); use mature form"),
    # warszawski2019
    ("d44_antigen",          "d44 antibody antigen (barnase or similar)",    True,  "check paper",                    "unknown",         "Verify from Warszawski 2019 paper; likely barnase P00967 or similar protein"),
    # zimmerman2020antibody
    ("fluorescein_zimm",     "Fluorescein (4-4-20 antibody target)",         False, "N/A (small molecule)",           "synthetic",       "Fluorescein is a small molecule (MW 332 Da); no protein sequence; antigen_source=missing is correct"),
    # adams2017measuring
    ("fluorescein_adams",    "Fluorescein (4-4-20 antibody target)",         False, "N/A (small molecule)",           "synthetic",       "Same as zimmerman; fluorescein has no sequence; antigen_source=missing is correct"),
    # garbinski2023
    ("garbinski_target",     "garbinski2023 kd target antigen",              True,  "check paper",                    "unknown",         "Verify antigen identity from Garbinski 2023 paper"),
    # shanker2024unsupervised
    ("SARS-CoV-2_SA58",      "SARS-CoV-2 Spike (SA58 antibody target)",     True,  "7LYL or variant",                "SARS-CoV-2",      "SA58 targets SARS-CoV-2 Spike; use variant-appropriate sequence"),
    ("SARS-CoV-2_Ly1404",    "SARS-CoV-2 Spike (Ly1404 antibody target)",   True,  "7LYL or variant",                "SARS-CoV-2",      "Ly1404 targets SARS-CoV-2 Spike; use variant-appropriate sequence"),
    # peterson2024integrated
    ("H1_HA_peterson",       "Influenza H1 HA (peterson integrated study)", True,  "check paper",                    "Influenza A",     "Verify specific H1 strain from Peterson 2024 paper"),
    # kirby2024retrospective
    ("CoV2_kirby",           "SARS-CoV-2 Spike (kirby retrospective)",      True,  "7LYL or RBD",                    "SARS-CoV-2",      "Verify whether full Spike or just RBD from Kirby 2024 paper"),
    # tsuruta
    ("CoV2_tsuruta",         "SARS-CoV-2 (tsuruta sarscov2 binary)",        True,  "7LYL or RBD",                    "SARS-CoV-2",      "Verify from Tsuruta 2024 paper; likely RBD or Spike ecto-domain"),
    # li2023machine / engelhart
    ("CoV2_RBD_li",          "SARS-CoV-2 RBD (li2023machine)",              True,  "6M0J",                           "SARS-CoV-2",      "PDB 6M0J chain E; RBD aa 319-541; or UniProt P0DTC2 aa 319-541"),
    ("CoV2_RBD_eng",         "SARS-CoV-2 RBD (engelhart2022dataset)",       True,  "6M0J",                           "SARS-CoV-2",      "Same as li2023machine; PDB 6M0J chain E"),
]

ANTIGEN_REFS_HEADER = [
    "antigen_key", "antigen_name", "is_protein",
    "likely_uniprot_or_pdb", "antigen_species", "retrieval_notes",
    "source_url",
]

# UniProt accessions are 6 chars: letter, digit, 3 alphanumeric, digit
# (e.g. P00698, Q9Y5U5). PDB IDs are 4 chars starting with a digit
# (e.g. 7LYL, 6M0J). Anything else (multiple candidates, "check paper",
# strain names, "N/A (small molecule)", etc.) cannot be mapped to a
# single record and is left blank.
_UNIPROT_RE = re.compile(r'^[A-Z][0-9][A-Z0-9]{3}[0-9]$')
_PDB_RE = re.compile(r'^[0-9][A-Za-z0-9]{3}$')


def _source_url(accession):
    """Derive a clickable source URL from likely_uniprot_or_pdb."""
    acc = accession.strip()
    if _UNIPROT_RE.match(acc):
        return f"https://www.uniprot.org/uniprotkb/{acc}/entry"
    if _PDB_RE.match(acc):
        return f"https://www.rcsb.org/structure/{acc}"
    return ""

# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    rows = []
    missing_antigens = {}   # antigen_key -> first dataset info

    for script_path in sorted(SCRIPTS_BASE.rglob("*/*/convert.py")):
        parts = script_path.parts
        # … /binding/{study_id}/{table_id}/convert.py
        try:
            idx_binding = [i for i, p in enumerate(parts) if p == "binding"][-1]
        except IndexError:
            continue
        study_id = parts[idx_binding + 1]
        table_id = parts[idx_binding + 2]

        text = script_path.read_text()
        source_file = _extract(text, "SOURCE_FILE")
        if not source_file:
            continue

        # Use csv_name = basename of source_file
        csv_name = Path(source_file).name

        antibody_type  = _extract(text, "ANTIBODY_TYPE", "unknown")
        antigen_key    = _extract(text, "ANTIGEN_KEY",   "unknown")
        antigen_name   = _extract(text, "ANTIGEN_NAME",  "")
        antigen_source = _extract(text, "ANTIGEN_SOURCE","missing")
        antigen_source_note = _extract(text, "ANTIGEN_SOURCE_NOTE", "")
        metric_name    = _extract(text, "METRIC_NAME",   "unknown")
        label_kind     = _extract(text, "LABEL_KIND",    "experimental")
        status         = "ready"
        notes          = _infer_notes(study_id, table_id, text, antigen_source, antigen_source_note, label_kind, source_file)

        # Special case: AbRank has two metrics, no simple constants
        if study_id == "AbRank":
            antigen_key   = "multi_antigen"
            antigen_name  = "multiple antigens (see Ag_name column)"
            metric_name   = "neg_log10_kd_M; neg_log10_ic50_ugml"
            label_kind    = "experimental"
            notes         = "multi-antigen; Kd+IC50 dual records; censored values dropped; zip streaming"

        rows.append(dict(
            study_id=study_id, table_id=table_id, csv_name=csv_name,
            antibody_type=antibody_type, antigen_key=antigen_key,
            antigen_name=antigen_name, antigen_source=antigen_source,
            metric_name=metric_name, label_kind=label_kind,
            status=status, notes=notes,
        ))

        if antigen_source in ("missing", "retrieved") and antigen_key not in missing_antigens:
            missing_antigens[antigen_key] = (antigen_name, study_id, table_id)

    # Add blocked datasets
    rows.extend(BLOCKED)

    # Sort: ready first, then blocked; alphabetically within
    rows.sort(key=lambda r: (0 if r["status"]=="ready" else 1, r["study_id"], r["table_id"]))

    # Write manifest.csv
    manifest_path = OUTDIR / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "study_id","table_id","csv_name","antibody_type",
            "antigen_key","antigen_name","antigen_source",
            "metric_name","label_kind","status","notes",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows → {manifest_path}")

    # Write antigen_missing_summary.csv from curated ANTIGEN_REFS
    missing_path = OUTDIR / "antigen_missing_summary.csv"
    with open(missing_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ANTIGEN_REFS_HEADER)
        w.writeheader()
        for ref in ANTIGEN_REFS:
            row = dict(zip(ANTIGEN_REFS_HEADER[:-1], ref))
            row["source_url"] = _source_url(row["likely_uniprot_or_pdb"])
            w.writerow(row)
    print(f"Wrote {len(ANTIGEN_REFS)} rows → {missing_path}")


if __name__ == "__main__":
    main()
