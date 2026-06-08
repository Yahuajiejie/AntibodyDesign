"""
MSA subpackage.

This package owns homolog FASTA handling and MSA construction utilities.
"""

from .builder import (
    build_mafft_command,
    read_a3m,
    sample_msa_depth,
    strip_a3m_insertions,
    validate_msa,
    write_a3m,
)
from .homolog_search import (
    FastaRecord,
    build_blastp_command,
    build_mmseqs_easy_search_command,
    deduplicate_records,
    filter_homologs,
    read_fasta,
    run_search_command,
    write_fasta,
    write_homolog_fasta,
)

__all__ = [
    "FastaRecord",
    "build_blastp_command",
    "build_mafft_command",
    "build_mmseqs_easy_search_command",
    "deduplicate_records",
    "filter_homologs",
    "read_a3m",
    "read_fasta",
    "run_search_command",
    "sample_msa_depth",
    "strip_a3m_insertions",
    "validate_msa",
    "write_a3m",
    "write_fasta",
    "write_homolog_fasta",
]
