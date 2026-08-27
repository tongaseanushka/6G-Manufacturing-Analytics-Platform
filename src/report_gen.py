"""
Report Generation CLI for Thales 6G Manufacturing Analytics Platform.

This script is the command-line entry point for generating the technical research
paper and executive summary reports from the Thales manufacturing dataset.

Usage:
    # Generate research paper
    python src/report_gen.py --output reports/research_paper.md

    # Generate executive summary
    python src/report_gen.py --summary --output reports/executive_summary.md

    # Specify a custom dataset path
    python src/report_gen.py --csv data/Thales_Group_Manufacturing.csv --output reports/research_paper.md

Requirements Addressed:
    - 15.1: Research paper CLI entry point
    - 16.1: Executive summary CLI entry point
    - 19.1–19.5: Deployment and reproducibility
    - 23.1–23.4: Reproducibility documentation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable regardless of CWD
sys.path.insert(0, str(Path(__file__).parent))

from data_prep import DataValidationError, load_and_prepare_dataset
from kpi_computation import compute_all_kpis
from paper_generator import generate_research_paper
from executive_summary_generator import generate_executive_summary


DEFAULT_CSV = Path(__file__).parent.parent / "data" / "Thales_Group_Manufacturing.csv"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="report_gen",
        description="Generate research paper or executive summary for the Thales 6G Analytics Platform.",
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help="Path to Thales_Group_Manufacturing.csv (default: data/Thales_Group_Manufacturing.csv)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output file path for the generated Markdown report.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate executive summary instead of full research paper.",
    )
    return parser.parse_args()


def main() -> None:
    """Main CLI entry point."""
    args = parse_args()

    csv_path = args.csv
    output_path = Path(args.output)
    generate_summary = args.summary

    # ---- Load data ----
    print(f"Loading dataset from: {csv_path}")
    try:
        df = load_and_prepare_dataset(csv_path)
        print(f"  [OK] Loaded {len(df):,} rows, {len(df.columns)} columns")
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        sys.exit(1)
    except DataValidationError as e:
        print(f"  [ERROR] Validation ERROR: {e}")
        sys.exit(1)

    # ---- Compute KPIs ----
    print("Computing KPIs...")
    kpis = compute_all_kpis(df)
    print(f"  [OK] NSI={kpis.nsi.nsi:.3f}, LSS={kpis.lss.score:.4f}, "
          f"PLIR={kpis.plir.ratio:.3f}, Cramér's V={kpis.nec.cramers_v:.3f}")

    # ---- Generate report ----
    if generate_summary:
        print("Generating executive summary...")
        paper = generate_research_paper(df, kpis)
        content = generate_executive_summary(paper, kpis, df)
        report_type = "Executive Summary"
    else:
        print("Generating research paper...")
        content = generate_research_paper(df, kpis)
        report_type = "Research Paper"

    # ---- Write output ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    word_count = len(content.split())
    print(f"  [OK] {report_type} written to: {output_path}")
    print(f"  [OK] Word count: {word_count:,}")

    if generate_summary and word_count > 900:
        print(f"  [WARN] Note: Summary is {word_count} words (target <= 800). Consider trimming.")

    print("\nDone.")


if __name__ == "__main__":
    main()
