"""Per-data-source noise-floor (tau) registry for `noise_aware_multiscale`.

`records.parquet` merges 8 originally separate data sources (different
assays, different labs, different eras) into one table, and -- critically --
does not preserve which source any given record came from as an explicit
column. What it does preserve is `antigen_key`, which turns out to almost
perfectly stand in for source: each source tested its antibodies against a
characteristic, disjoint set of antigens (AlphaSeq only against
`SARS_CoV_2`, CATNAP only against `HIV_*` strains, RBD-escape's Kd-bearing
subset only against individual point-mutant antigens named
`SARS_CoV_2_<mutation>`, etc.). Matching by record count against the raw
`AbRank_dataset.csv.zip`'s `Source` column confirms this correspondence to
within 0.5% -- see `docs/experiments/tau_registry.md` for the full
cross-check table and every rule's literature basis.

The table below is hardcoded, not YAML-configurable: it encodes facts looked
up from the literature about each assay's measurement reproducibility, not
a per-run hyperparameter a config file should casually override. A caller
that genuinely needs a different tau for a specific run can still pass an
explicit `default_tau` override for anything this registry doesn't match,
or call `_noise_aware_multiscale_pairs` directly with its own tau.

Confidence varies a lot across rules -- see the module docstring's citations
and `docs/experiments/tau_registry.md` §3 for the honest version of "how
sure are we": AlphaSeq's 0.3 was validated against real data in a prior
analysis; RBD-escape's 0.15 and CATNAP's 0.5 are derived from a related but
not directly equivalent published statistic (R², a fold-difference
acceptance criterion) rather than a directly reported log10(Kd) standard
deviation; SKEMPIv2's 0.35 is a unit conversion from a ddG distribution
width, not a measurement-error figure specifically; and `_DEFAULT_TAU`
covers everything else (well under 1% of records) with no literature
backing at all, purely so an unmatched group degrades gracefully instead of
crashing or silently using `tau=0`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

_DEFAULT_TAU = 0.2
_DEFAULT_LABEL = "default_unresearched"
_DEFAULT_BASIS = (
    "No source-specific reproducibility research done (combined record "
    "share < 1%); a conservative placeholder, not a researched value -- "
    "see docs/experiments/tau_registry.md §3."
)


@dataclass(frozen=True)
class _TauRule:
    label: str
    pattern: re.Pattern[str]
    tau: float
    basis: str


_RULES: tuple[_TauRule, ...] = (
    _TauRule(
        label="alphaseq_sars_cov_2",
        pattern=re.compile(r"^SARS_CoV_2$"),
        tau=0.3,
        basis=(
            "Engelhart et al., Sci Data 9:653 (2022) -- AlphaSeq SARS-CoV-2 "
            "Kd dataset. Inter-replicate Pearson r=0.66-0.93 (triplicate "
            "measurements); k=1-mutant affinity IQR=1.69 log10(nM). "
            "Estimated true measurement noise ~0.47-1.03 log10 units; 0.3 "
            "is conservative relative to that range. Validated against real "
            "data in docs/experiments/noise_floor_tree_analysis.md."
        ),
    ),
    _TauRule(
        label="bloom_titeseq_sarbecovirus",
        pattern=re.compile(
            r"^(SARS_CoV_2_.+|SARS_CoV|WIV1_CoV|SHC014_CoV|MERS_CoV)$"
        ),
        tau=0.15,
        basis=(
            "Starr et al., Cell 182:1295 (2020) and the Omicron follow-up "
            "(Taylor et al., PLoS Pathog 2022) -- Bloom lab yeast-display "
            "Titeseq, same method family used across the point-mutant RBD "
            "panels and the cross-clade sarbecovirus panel. Reported "
            "duplicate-library R^2 > 0.99. No directly reported log10(Kd) "
            "replicate SD was found; 0.15 is back-derived from R^2 assuming "
            "a ~1-1.5 log10-unit total affinity spread across the scan, so "
            "this is weaker evidence than the AlphaSeq entry."
        ),
    ),
    _TauRule(
        label="catnap_hiv_neutralization",
        pattern=re.compile(r"^HIV_"),
        tau=0.5,
        basis=(
            "CATNAP aggregates literature IC50 values across many labs/eras "
            "with no single protocol, so no single reproducibility figure "
            "applies directly. Closest quantitative anchor: the standardized "
            "TZM-bl/A3R5 consortium validation (Sarzotti-Kelsoe et al., J "
            "Immunol Methods 2014; A3R5 validation, PMC4138262) accepts "
            "repeatability/intermediate-precision/reproducibility within a "
            "3-fold boundary (~0.48 log10). That is a best-case, "
            "single-protocol number; CATNAP's true cross-study noise is "
            "likely larger, so 0.5 is only a slight margin above that floor, "
            "not a confident upper bound."
        ),
    ),
    _TauRule(
        label="skempi_ddg",
        pattern=re.compile(r"^AgSKEMPI"),
        tau=0.35,
        basis=(
            "SKEMPI 2.0 (Jankauskaite et al., Bioinformatics 35:462, 2019): "
            "repeated measurements of the same mutation can disagree by on "
            "the order of 0.5 kcal/mol in ddG. Converting via "
            "ddG = -RT*ln(Kd_mut/Kd_wt) at 298K (RT*ln10 ~= 1.365 kcal/mol) "
            "gives ~0.37 log10 units; rounded to 0.35. This is a unit "
            "conversion from a distribution-width statistic, not a "
            "purpose-reported measurement-error figure."
        ),
    ),
)


def resolve_tau_for_group(
    group: pd.DataFrame, default_tau: float = _DEFAULT_TAU
) -> tuple[float, str, str]:
    """Look up `(tau, rule_label, basis)` for one group via its `antigen_key`.

    Every record in a well-formed group shares one `antigen_key` (group_id
    already encodes it, see `schema.py`'s group_id format) -- this is
    asserted, not silently assumed, so a malformed/mixed group fails loudly
    rather than picking an arbitrary member's antigen.

    Args:
        group: One group's trainable records; must have an `antigen_key`
            column with exactly one distinct value.
        default_tau: Fallback tau for any `antigen_key` none of the
            hardcoded rules match. Overridable by the caller (e.g. from
            `DataConfig.noise_aware_default_tau`) -- everything else in this
            registry is not, see module docstring.

    Returns:
        `(tau, rule_label, basis)`. `rule_label` is `"default_unresearched"`
        and `basis` explains why when no rule matched.

    Raises:
        ValueError: If `group` has zero or more than one distinct
            `antigen_key`.
    """
    antigen_keys = group["antigen_key"].astype(str).unique()
    if len(antigen_keys) != 1:
        raise ValueError(
            "resolve_tau_for_group expects a single-antigen group "
            f"(group_id already encodes antigen_key), got {len(antigen_keys)} "
            f"distinct antigen_key values: {sorted(antigen_keys)[:5]}"
        )
    antigen_key = antigen_keys[0]

    for rule in _RULES:
        if rule.pattern.match(antigen_key):
            return rule.tau, rule.label, rule.basis
    return default_tau, _DEFAULT_LABEL, _DEFAULT_BASIS
