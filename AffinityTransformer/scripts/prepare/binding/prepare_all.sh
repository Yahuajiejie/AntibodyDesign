#!/usr/bin/env bash
# Run all 'ready' binding dataset prepare scripts from the repo root.
#
# Usage (from repo root):
#   bash scripts/prepare/binding/prepare_all.sh
#   bash scripts/prepare/binding/prepare_all.sh 2>&1 | tee prepare_all.log
#
# Each per-dataset prepare.sh:
#   1. Runs convert.py  → processed/binding/{study_id}/{table_id}/records.parquet
#   2. Validates schema via validate_processed_table.py
#
# 'blocked' datasets are NOT run here.
# See manifest.csv for full dataset list and status.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

# ── Bootstrap: regenerate manifest + antigen_missing_summary ─────────────────
echo "[prepare_all] Generating metadata files ..."
python3 scripts/prepare/binding/gen_manifest.py

mkdir -p processed/binding
cp scripts/prepare/binding/manifest.csv           processed/binding/manifest.csv
cp scripts/prepare/binding/antigen_missing_summary.csv \
                                                  processed/binding/antigen_missing_summary.csv
echo "[prepare_all] Metadata files ready in processed/binding/"

PASS=0
FAIL=0
FAILED_DATASETS=()

run_one() {
    local script="$1"
    local label
    label="$(dirname "$script" | sed 's|scripts/prepare/binding/||')"
    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  $label"
    echo "══════════════════════════════════════════════════"
    if bash "$script"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        FAILED_DATASETS+=("$label")
        echo "  [ERROR] $label failed"
    fi
}

# ── Ready datasets (84 = 39 original + 45 new) ───────────────────────────────

# phillips2021binding
run_one scripts/prepare/binding/phillips2021binding/cr6261_h1_kd/prepare.sh
run_one scripts/prepare/binding/phillips2021binding/cr6261_h9_kd/prepare.sh
run_one scripts/prepare/binding/phillips2021binding/cr9114_h1_kd/prepare.sh
run_one scripts/prepare/binding/phillips2021binding/cr9114_h3_kd/prepare.sh

# koenig2017mutational
run_one scripts/prepare/binding/koenig2017mutational/kd_g6/prepare.sh

# hie2023efficient
run_one scripts/prepare/binding/hie2023efficient/CoV2Beta_C143_Kd/prepare.sh
run_one scripts/prepare/binding/hie2023efficient/CoV2Beta_REGN10987_Kd/prepare.sh
run_one scripts/prepare/binding/hie2023efficient/CoV2_S309_Kd/prepare.sh
run_one scripts/prepare/binding/hie2023efficient/CoV2omicron_REGN10987_Kd/prepare.sh
run_one scripts/prepare/binding/hie2023efficient/MEDIUCA_H1Solomon_Kd/prepare.sh
run_one scripts/prepare/binding/hie2023efficient/MEDIUCA_H4Hubei_Kd/prepare.sh
run_one scripts/prepare/binding/hie2023efficient/MEDI_H4Hubei_Kd/prepare.sh
run_one scripts/prepare/binding/hie2023efficient/MEDI_H7HK16_Kd/prepare.sh
run_one scripts/prepare/binding/hie2023efficient/ebola_mab114_Kd/prepare.sh

# hutchinson2023enhancement
run_one scripts/prepare/binding/hutchinson2023enhancement/multikd_fab/prepare.sh
run_one scripts/prepare/binding/hutchinson2023enhancement/multikd_igg/prepare.sh
run_one scripts/prepare/binding/hutchinson2023enhancement/singlekd_fab/prepare.sh
run_one scripts/prepare/binding/hutchinson2023enhancement/singlekd_igg/prepare.sh
run_one scripts/prepare/binding/hutchinson2023enhancement/top200kd_fab/prepare.sh
run_one scripts/prepare/binding/hutchinson2023enhancement/top200kd_igg/prepare.sh
run_one scripts/prepare/binding/hutchinson2023enhancement/top27kd_fab/prepare.sh
run_one scripts/prepare/binding/hutchinson2023enhancement/top27kd_igg/prepare.sh

# jain2024assessment
run_one scripts/prepare/binding/jain2024assessment/Hen_Lys_kd/prepare.sh
run_one scripts/prepare/binding/jain2024assessment/mouse_Ly_kd/prepare.sh

# rosace2023automated
run_one scripts/prepare/binding/rosace2023automated/kd_adalimumab/prepare.sh
run_one scripts/prepare/binding/rosace2023automated/kd_golimumab/prepare.sh

# shanehsazzadeh2023unlocking
run_one scripts/prepare/binding/shanehsazzadeh2023unlocking/kd_hher2_fab/prepare.sh
run_one scripts/prepare/binding/shanehsazzadeh2023unlocking/kd_hher2_mab/prepare.sh
run_one scripts/prepare/binding/shanehsazzadeh2023unlocking/zerokd_trastuzumab/prepare.sh

# shanehsazzadeh2024igdesign
run_one scripts/prepare/binding/shanehsazzadeh2024igdesign/Afasevikumab-IL17A_kd/prepare.sh
run_one scripts/prepare/binding/shanehsazzadeh2024igdesign/Bimagrumab-ACVR2B_kd/prepare.sh
run_one scripts/prepare/binding/shanehsazzadeh2024igdesign/Eculizumab-C5_kd/prepare.sh
run_one scripts/prepare/binding/shanehsazzadeh2024igdesign/Osocimab-FXI_kd/prepare.sh
run_one scripts/prepare/binding/shanehsazzadeh2024igdesign/Spesolimab-IL36R_kd/prepare.sh
run_one scripts/prepare/binding/shanehsazzadeh2024igdesign/Tezepelumab-TSLP_kd/prepare.sh
run_one scripts/prepare/binding/shanehsazzadeh2024igdesign/Utomilumab-TNFRSF9_kd/prepare.sh

# warszawski2019
run_one scripts/prepare/binding/warszawski2019/d44_Kd/prepare.sh

# zimmerman2020antibody
run_one scripts/prepare/binding/zimmerman2020antibody/4420_kd/prepare.sh

# garbinski2023
run_one scripts/prepare/binding/garbinski2023/kd/prepare.sh

# ── New datasets (45) — previously planned, now ready ────────────────────────

# kothiwal2025htp (10 EC50 + 10 SPR = 20)
run_one scripts/prepare/binding/kothiwal2025htp/DCC_ec50/prepare.sh
run_one scripts/prepare/binding/kothiwal2025htp/DCC_spr/prepare.sh
run_one scripts/prepare/binding/kothiwal2025htp/DKK_1.00_ec50/prepare.sh
run_one scripts/prepare/binding/kothiwal2025htp/DKK_1.00_spr/prepare.sh
run_one scripts/prepare/binding/kothiwal2025htp/IL23R_ec50/prepare.sh
run_one scripts/prepare/binding/kothiwal2025htp/IL23R_spr/prepare.sh
run_one scripts/prepare/binding/kothiwal2025htp/LOX1_ec50/prepare.sh
run_one scripts/prepare/binding/kothiwal2025htp/LOX1_spr/prepare.sh
run_one scripts/prepare/binding/kothiwal2025htp/PDL1_ec50/prepare.sh
run_one scripts/prepare/binding/kothiwal2025htp/PDL1_spr/prepare.sh
run_one scripts/prepare/binding/kothiwal2025htp/PDL2_ec50/prepare.sh
run_one scripts/prepare/binding/kothiwal2025htp/PDL2_spr/prepare.sh
run_one scripts/prepare/binding/kothiwal2025htp/ROBO1_ec50/prepare.sh
run_one scripts/prepare/binding/kothiwal2025htp/ROBO1_spr/prepare.sh
run_one scripts/prepare/binding/kothiwal2025htp/ROBO2N_hROBO2N_ec50/prepare.sh
run_one scripts/prepare/binding/kothiwal2025htp/ROBO2N_hROBO2N_spr/prepare.sh
run_one scripts/prepare/binding/kothiwal2025htp/Syncytin2_ec50/prepare.sh
run_one scripts/prepare/binding/kothiwal2025htp/Syncytin2_spr/prepare.sh
run_one scripts/prepare/binding/kothiwal2025htp/TIGIT_ec50/prepare.sh
run_one scripts/prepare/binding/kothiwal2025htp/TIGIT_spr/prepare.sh

# shanker2024unsupervised (7)
run_one scripts/prepare/binding/shanker2024unsupervised/Ly1404-BQ.1.1_IC50/prepare.sh
run_one scripts/prepare/binding/shanker2024unsupervised/Ly1404-BQ.1.1_Kd/prepare.sh
run_one scripts/prepare/binding/shanker2024unsupervised/Ly1404_Wuhan_IC50/prepare.sh
run_one scripts/prepare/binding/shanker2024unsupervised/SA58-BA.1_IC50/prepare.sh
run_one scripts/prepare/binding/shanker2024unsupervised/SA58-BQ.1.1_IC50/prepare.sh
run_one scripts/prepare/binding/shanker2024unsupervised/SA58-BQ.1.1_Kd/prepare.sh
run_one scripts/prepare/binding/shanker2024unsupervised/SA58-XBB.1.5_Kd/prepare.sh

# adams2017measuring (2)
run_one scripts/prepare/binding/adams2017measuring/4420-fluorescein_kd-flow/prepare.sh
run_one scripts/prepare/binding/adams2017measuring/4420-fluorescein_kd-titeseq/prepare.sh

# makowski2022cooptimization (4; iso_ant source file has known filename typo)
run_one scripts/prepare/binding/makowski2022cooptimization/igg_ant/prepare.sh
run_one scripts/prepare/binding/makowski2022cooptimization/igg_ova/prepare.sh
run_one scripts/prepare/binding/makowski2022cooptimization/iso_ant/prepare.sh
run_one scripts/prepare/binding/makowski2022cooptimization/iso_ova/prepare.sh

# peterson2024integrated (2)
run_one scripts/prepare/binding/peterson2024integrated/ab_H1HA_binary/prepare.sh
run_one scripts/prepare/binding/peterson2024integrated/ab_H1HA_kd/prepare.sh

# kirby2024retrospective (2)
run_one scripts/prepare/binding/kirby2024retrospective/ab-SARSCoV2_binary_kd/prepare.sh
run_one scripts/prepare/binding/kirby2024retrospective/ab-SARSCoV2_kd/prepare.sh

# cognano (1)
run_one scripts/prepare/binding/cognano/AVIDa-hTNFa/prepare.sh

# shanehsazzadeh2023unlocking — adcc_ec50 (1; others already in ready block above)
run_one scripts/prepare/binding/shanehsazzadeh2023unlocking/adcc_ec50/prepare.sh

# li2023machine — zip, ~700 MB each; streaming parquet writes (2)
run_one scripts/prepare/binding/li2023machine/affinity1/prepare.sh
run_one scripts/prepare/binding/li2023machine/affinity2/prepare.sh

# engelhart2022dataset — zip (1)
run_one scripts/prepare/binding/engelhart2022dataset/scFv-SARS-CoV-2_affinity/prepare.sh

# tsuruta — zip, VHH signal-peptide + His-tag stripped (2)
run_one scripts/prepare/binding/tsuruta2024sarscov2/binary/prepare.sh
run_one scripts/prepare/binding/tsuruta2024avida/hIL6_binary/prepare.sh

# AbRank — zip, multi-metric (Kd + IC50), streaming (1)
run_one scripts/prepare/binding/AbRank/dataset/prepare.sh

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo "  SUMMARY:  pass=$PASS  fail=$FAIL"
if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    echo "  FAILED:"
    for d in "${FAILED_DATASETS[@]}"; do
        echo "    - $d"
    done
fi
echo "══════════════════════════════════════════════════"

[ "$FAIL" -eq 0 ]
