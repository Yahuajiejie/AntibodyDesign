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
    ("Fluorescein",          "Fluorescein (small molecule hapten)",          False, "N/A (small molecule)",           "synthetic",       "No protein sequence; keep antigen_source=missing unless a small-molecule featurizer is added"),
    ("unknown_antigen",      "Unknown proprietary antigen",                  True,  "check paper",                    "unknown",         "Verify antigen identity from Garbinski 2023 paper/source before adding any sequence"),
    ("H1_HA",                "Influenza A H1 hemagglutinin",                 True,  "check paper",                    "Influenza A",     "Peterson processed tables do not expose the strain; fill only after exact H1HA strain/construct is confirmed"),
    ("CoV2_Beta_S2P",        "SARS-CoV-2 Beta variant Spike S2P",            True,  "variant-specific sequence",      "SARS-CoV-2",      "Need exact S2P/Beta construct sequence before filling; do not reuse Wuhan RBD silently"),
    ("CoV2_Omicron_RBD",     "SARS-CoV-2 Omicron RBD",                       True,  "variant-specific sequence",      "SARS-CoV-2",      "Need exact Omicron lineage/construct sequence before filling"),
    ("H4_Hubei_HA",          "Influenza A H4 HA (Hubei)",                    True,  "check strain accession",         "Influenza A",     "Need exact H4 Hubei strain and construct accession from Hie 2023 supplementary/source"),
    ("H7_HK16_HA",           "Influenza A H7 HA (HK16)",                     True,  "check strain accession",         "Influenza A",     "Need exact H7 HK16 strain and construct accession from Hie 2023 supplementary/source"),
    ("CoV2_BA1_Spike",       "SARS-CoV-2 BA.1 Spike",                        True,  "variant-specific sequence",      "SARS-CoV-2",      "Need exact BA.1 construct for Shanker 2024 before filling"),
    ("CoV2_BQ11_Spike",      "SARS-CoV-2 BQ.1.1 Spike",                      True,  "variant-specific sequence",      "SARS-CoV-2",      "Need exact BQ.1.1 construct for Shanker 2024 before filling"),
    ("CoV2_XBB15_Spike",     "SARS-CoV-2 XBB.1.5 Spike",                     True,  "variant-specific sequence",      "SARS-CoV-2",      "Need exact XBB.1.5 construct for Shanker 2024 before filling"),
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
            notes         = (
                "multi-antigen; Kd+IC50 dual records; censored values dropped; "
                "zip streaming; SARS-CoV-2 RBD single mutants derived from UniProt "
                "P0DTC2 aa319-541 using AbRank position + 1"
            )

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
