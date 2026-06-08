"""
registry/build.py — antigen_registry 命令行入口

推荐从仓库根目录运行：

  python -m FLAb.AffinityTransformer.registry.build
"""

from __future__ import annotations

import argparse

from .workflow import (
    build_registry_from_paths,
    write_registry_result,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """
    构造命令行参数解析器。

    输入：
      无。

    返回：
      argparse.ArgumentParser。
    """
    parser = argparse.ArgumentParser(
        description="Build AffinityTransformer antigen_registry.csv",
    )
    parser.add_argument(
        "--binding_dir",
        default="FLAb/data/binding",
        help="FLAb binding CSV/CSV.zip directory; use empty string to skip",
    )
    parser.add_argument(
        "--metadata_csv",
        default="FLAb/data/flab_metadata.csv",
        help="FLAb metadata CSV; use empty string to skip",
    )
    parser.add_argument(
        "--binding_pattern",
        default="*.csv*",
        help="glob pattern inside binding_dir",
    )
    parser.add_argument(
        "--max_binding_files",
        type=int,
        default=None,
        help="debug only: read at most N binding files",
    )
    parser.add_argument(
        "--tasks_md",
        default="references/task_docs/TASKS.md",
        help="official task markdown; use empty string to skip",
    )
    parser.add_argument(
        "--proteinbase_csv",
        default="competition_data/_Preliminary_SequenceData/22/proteinbase_all_data_28_01_2026.csv",
        help="proteinbase CSV; use empty string to skip",
    )
    parser.add_argument(
        "--andd_xlsx",
        default="competition_data/_Preliminary_NanobodyData/ANDD.xlsx",
        help="ANDD xlsx; use empty string to skip",
    )
    parser.add_argument(
        "--sabdab_summary_tsv",
        default="competition_data/_Preliminary_StructureData/sabdab_summary_all.tsv",
        help="SAbDab summary TSV; use empty string to skip",
    )
    parser.add_argument(
        "--output",
        default="FLAb/results/v3/registry/antigen_registry.csv",
        help="output antigen registry CSV path",
    )
    parser.add_argument(
        "--issues_output",
        default="FLAb/results/v3/registry/antigen_registry_issues.csv",
        help="output registry validation report path",
    )
    parser.add_argument(
        "--auxiliary_dir",
        default="FLAb/results/v3/auxiliary",
        help="directory for task_controls/proteinbase_targets tables",
    )
    parser.add_argument(
        "--skip_external",
        action="store_true",
        help="only build registry from FLAb binding CSV/metadata",
    )
    return parser


def _none_if_empty(value: str | None) -> str | None:
    """将命令行中的空字符串转成 None。"""
    if value is None:
        return None
    value = value.strip()
    return value or None


def main() -> None:
    """
    CLI 主函数。

    输入：
      命令行参数。

    输出：
      写出 antigen_registry、质检报告和辅助表；在 stdout 打印摘要。
    """
    args = build_arg_parser().parse_args()
    result = build_registry_from_paths(
        binding_dir=_none_if_empty(args.binding_dir),
        metadata_csv=_none_if_empty(args.metadata_csv),
        binding_pattern=args.binding_pattern,
        max_binding_files=args.max_binding_files,
        tasks_md=_none_if_empty(args.tasks_md),
        proteinbase_csv=_none_if_empty(args.proteinbase_csv),
        andd_xlsx=_none_if_empty(args.andd_xlsx),
        sabdab_summary_tsv=_none_if_empty(args.sabdab_summary_tsv),
        include_external=not args.skip_external,
    )
    written = write_registry_result(
        result=result,
        registry_path=args.output,
        issues_path=_none_if_empty(args.issues_output),
        auxiliary_dir=_none_if_empty(args.auxiliary_dir),
    )

    print("[registry] antigen registry rows:", len(result.registry))
    print("[registry] registry groups:", result.registry["compatible_group"].nunique())
    if not result.issues.empty:
        print("[registry] validation issues:", len(result.issues))
        print(result.issues["severity"].value_counts().to_string())
    else:
        print("[registry] validation issues: 0")
    for name, path in written.items():
        print(f"[registry] wrote {name}: {path}")


if __name__ == "__main__":
    main()
