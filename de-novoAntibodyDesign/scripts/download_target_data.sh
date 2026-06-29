#!/usr/bin/env bash
set -euo pipefail

# Fill these before running.
UNIPROT_ID="P00000"
PDB_IDS=("XXXX")

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${ROOT_DIR}/data/raw"

mkdir -p "${RAW_DIR}/sequence" "${RAW_DIR}/structures" "${RAW_DIR}/variants" "${RAW_DIR}/literature"

echo "Downloading UniProt FASTA for ${UNIPROT_ID}"
curl -L "https://rest.uniprot.org/uniprotkb/${UNIPROT_ID}.fasta" -o "${RAW_DIR}/sequence/${UNIPROT_ID}.fasta"

echo "Downloading AlphaFold DB model for ${UNIPROT_ID}"
curl -L "https://alphafold.ebi.ac.uk/files/AF-${UNIPROT_ID}-F1-model_v4.cif" -o "${RAW_DIR}/structures/AF-${UNIPROT_ID}-F1-model_v4.cif"

for PDB_ID in "${PDB_IDS[@]}"; do
  LOWER_ID="$(echo "${PDB_ID}" | tr '[:upper:]' '[:lower:]')"
  UPPER_ID="$(echo "${PDB_ID}" | tr '[:lower:]' '[:upper:]')"
  echo "Downloading RCSB PDB mmCIF for ${UPPER_ID}"
  curl -L "https://files.rcsb.org/download/${UPPER_ID}.cif" -o "${RAW_DIR}/structures/${LOWER_ID}.cif"
done

echo "Done. Check files under ${RAW_DIR}"

