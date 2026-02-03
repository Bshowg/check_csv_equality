#!/usr/bin/env python3
"""
analyze_asset_portafoglio.py

Deep analysis of differences between two AssetPortafoglio.csv files.
Performs detailed comparison including:
- Row matching by ISIN key
- Column-by-column comparison
- Numeric precision differences
- Missing/extra rows
- Summary statistics

Usage:
  python -m src.analyze_asset_portafoglio data/og/Attivi/AssetPortafoglio.csv data/gen/Attivi/AssetPortafoglio.csv
  python -m src.analyze_asset_portafoglio data/og/Attivi/AssetPortafoglio.csv data/gen/Attivi/AssetPortafoglio.csv --json report.json
"""

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Any


@dataclass
class ColumnDifference:
    column_name: str
    value_og: str
    value_gen: str
    difference_type: str  # 'exact', 'numeric', 'whitespace', 'missing'
    numeric_diff: Optional[float] = None


@dataclass
class RowComparison:
    key: str  # The ISIN or identifier
    idasset_og: Optional[str] = None
    idasset_gen: Optional[str] = None
    status: str = "matched"  # 'matched', 'only_og', 'only_gen'
    differences: List[ColumnDifference] = None
    
    def __post_init__(self):
        if self.differences is None:
            self.differences = []


@dataclass
class SummaryStats:
    total_rows_og: int
    total_rows_gen: int
    matched_rows: int
    only_in_og: int
    only_in_gen: int
    rows_with_differences: int
    total_column_differences: int
    columns_with_differences: Dict[str, int]  # column_name -> count


def read_csv_as_dict(
    path: Path,
    delimiter: str = ";",
    encoding: str = "utf-8-sig",
    key_columns: List[str] = None
) -> tuple[List[str], Dict[str, Dict[str, str]]]:
    """
    Read CSV and return headers and a dictionary keyed by specified columns.
    Returns: (headers, {key: {column: value}})
    """
    if key_columns is None:
        key_columns = ["ISIN"]
    
    rows_dict = {}
    
    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        headers = reader.fieldnames or []
        
        for row in reader:
            # Build composite key from multiple columns
            key_parts = [row.get(col, "") for col in key_columns]
            key = "|".join(key_parts)
            
            if key and key != "|" * (len(key_columns) - 1):  # Not all empty
                # Handle duplicate keys by appending a counter
                if key in rows_dict:
                    counter = 1
                    original_key = key
                    while key in rows_dict:
                        key = f"{original_key}_dup{counter}"
                        counter += 1
                rows_dict[key] = row
    
    return headers, rows_dict


def is_numeric(value: str) -> bool:
    """Check if a string represents a number."""
    try:
        float(value.replace(",", "."))
        return True
    except (ValueError, AttributeError):
        return False


def normalize_numeric(value: str) -> Optional[float]:
    """Convert numeric string to float."""
    try:
        return float(value.replace(",", "."))
    except (ValueError, AttributeError):
        return None


def compare_values(
    col_name: str,
    val_og: str,
    val_gen: str,
    numeric_tolerance: float = 1e-6
) -> Optional[ColumnDifference]:
    """
    Compare two values and return a ColumnDifference if they differ.
    """
    # Both empty or None
    if not val_og and not val_gen:
        return None
    
    # Exact match
    if val_og == val_gen:
        return None
    
    # Check whitespace differences
    if val_og.strip() == val_gen.strip():
        return ColumnDifference(
            column_name=col_name,
            value_og=val_og,
            value_gen=val_gen,
            difference_type="whitespace"
        )
    
    # Check numeric differences
    if is_numeric(val_og) and is_numeric(val_gen):
        num_og = normalize_numeric(val_og)
        num_gen = normalize_numeric(val_gen)
        
        if num_og is not None and num_gen is not None:
            diff = abs(num_og - num_gen)
            if diff < numeric_tolerance:
                return None  # Within tolerance
            
            return ColumnDifference(
                column_name=col_name,
                value_og=val_og,
                value_gen=val_gen,
                difference_type="numeric",
                numeric_diff=diff
            )
    
    # Different values
    return ColumnDifference(
        column_name=col_name,
        value_og=val_og,
        value_gen=val_gen,
        difference_type="exact"
    )


def analyze_files(
    path_og: Path,
    path_gen: Path,
    delimiter: str = ";",
    encoding: str = "utf-8-sig",
    key_columns: List[str] = None,
    numeric_tolerance: float = 1e-6,
    exclude_columns: Optional[List[str]] = None
) -> tuple[List[RowComparison], SummaryStats]:
    """
    Analyze differences between two CSV files.
    """
    if key_columns is None:
        key_columns = ["ISIN"]
    if exclude_columns is None:
        exclude_columns = []
    
    # Read both files
    headers_og, data_og = read_csv_as_dict(path_og, delimiter, encoding, key_columns)
    headers_gen, data_gen = read_csv_as_dict(path_gen, delimiter, encoding, key_columns)
    
    # Get all unique keys
    all_keys = set(data_og.keys()) | set(data_gen.keys())
    
    comparisons: List[RowComparison] = []
    columns_with_diffs: Dict[str, int] = {}
    rows_with_diffs = 0
    
    for key in sorted(all_keys):
        row_og = data_og.get(key)
        row_gen = data_gen.get(key)
        
        comparison = RowComparison(key=key)
        
        # Only in og
        if row_og and not row_gen:
            comparison.status = "only_og"
            comparison.idasset_og = row_og.get("idasset", "")
            comparisons.append(comparison)
            continue
        
        # Only in gen
        if row_gen and not row_og:
            comparison.status = "only_gen"
            comparison.idasset_gen = row_gen.get("idasset", "")
            comparisons.append(comparison)
            continue
        
        # Both exist - compare columns
        comparison.idasset_og = row_og.get("idasset", "")
        comparison.idasset_gen = row_gen.get("idasset", "")
        
        # Compare all columns
        all_columns = set(headers_og) | set(headers_gen)
        
        for col in all_columns:
            if col in exclude_columns:
                continue
            
            val_og = row_og.get(col, "")
            val_gen = row_gen.get(col, "")
            
            diff = compare_values(col, val_og, val_gen, numeric_tolerance)
            if diff:
                comparison.differences.append(diff)
                columns_with_diffs[col] = columns_with_diffs.get(col, 0) + 1
        
        if comparison.differences:
            rows_with_diffs += 1
        
        comparisons.append(comparison)
    
    # Calculate summary statistics
    stats = SummaryStats(
        total_rows_og=len(data_og),
        total_rows_gen=len(data_gen),
        matched_rows=len([c for c in comparisons if c.status == "matched"]),
        only_in_og=len([c for c in comparisons if c.status == "only_og"]),
        only_in_gen=len([c for c in comparisons if c.status == "only_gen"]),
        rows_with_differences=rows_with_diffs,
        total_column_differences=sum(len(c.differences) for c in comparisons),
        columns_with_differences=columns_with_diffs
    )
    
    return comparisons, stats


def print_human_report(
    comparisons: List[RowComparison],
    stats: SummaryStats,
    max_rows_to_show: int = 50,
    show_all_differences: bool = False
) -> None:
    """
    Print a human-readable report.
    """
    print("=" * 100)
    print("ASSETPORTAFOGLIO COMPARISON REPORT")
    print("=" * 100)
    
    print("\nSUMMARY STATISTICS:")
    print(f"  Total rows in OG:  {stats.total_rows_og}")
    print(f"  Total rows in GEN: {stats.total_rows_gen}")
    print(f"  Matched rows:      {stats.matched_rows}")
    print(f"  Only in OG:        {stats.only_in_og}")
    print(f"  Only in GEN:       {stats.only_in_gen}")
    print(f"  Rows with diffs:   {stats.rows_with_differences}")
    print(f"  Total column diffs: {stats.total_column_differences}")
    
    if stats.columns_with_differences:
        print("\nCOLUMNS WITH DIFFERENCES (sorted by frequency):")
        sorted_cols = sorted(
            stats.columns_with_differences.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for col, count in sorted_cols[:20]:
            print(f"  {col:40s}: {count:4d} rows")
        
        if len(sorted_cols) > 20:
            print(f"  ... and {len(sorted_cols) - 20} more columns")
    
    # Show rows only in OG
    only_og = [c for c in comparisons if c.status == "only_og"]
    if only_og:
        print(f"\nROWS ONLY IN OG ({len(only_og)} total):")
        for comp in only_og[:10]:
            print(f"  ISIN: {comp.key:30s} (idasset: {comp.idasset_og})")
        if len(only_og) > 10:
            print(f"  ... and {len(only_og) - 10} more rows")
    
    # Show rows only in GEN
    only_gen = [c for c in comparisons if c.status == "only_gen"]
    if only_gen:
        print(f"\nROWS ONLY IN GEN ({len(only_gen)} total):")
        for comp in only_gen[:10]:
            print(f"  ISIN: {comp.key:30s} (idasset: {comp.idasset_gen})")
        if len(only_gen) > 10:
            print(f"  ... and {len(only_gen) - 10} more rows")
    
    # Show rows with differences
    rows_with_diffs = [c for c in comparisons if c.differences]
    if rows_with_diffs:
        print(f"\nROWS WITH DIFFERENCES ({len(rows_with_diffs)} total, showing first {min(max_rows_to_show, len(rows_with_diffs))}):")
        
        for i, comp in enumerate(rows_with_diffs[:max_rows_to_show]):
            print(f"\n  [{i+1}] ISIN: {comp.key}")
            print(f"      idasset: OG={comp.idasset_og}, GEN={comp.idasset_gen}")
            print(f"      Differences: {len(comp.differences)} columns")
            
            if show_all_differences or len(comp.differences) <= 10:
                for diff in comp.differences:
                    if diff.difference_type == "numeric":
                        print(f"        - {diff.column_name}:")
                        print(f"            OG:  {diff.value_og}")
                        print(f"            GEN: {diff.value_gen}")
                        print(f"            DIFF: {diff.numeric_diff:.10f}")
                    else:
                        print(f"        - {diff.column_name} ({diff.difference_type}):")
                        print(f"            OG:  {diff.value_og}")
                        print(f"            GEN: {diff.value_gen}")
            else:
                print(f"        (showing first 5 of {len(comp.differences)} differences)")
                for diff in comp.differences[:5]:
                    print(f"        - {diff.column_name} ({diff.difference_type})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deep analysis of AssetPortafoglio.csv differences"
    )
    parser.add_argument("file_og", type=str, help="Original file path")
    parser.add_argument("file_gen", type=str, help="Generated file path")
    parser.add_argument("--delimiter", default=";", help="CSV delimiter (default: ;)")
    parser.add_argument("--encoding", default="utf-8-sig", help="File encoding (default: utf-8-sig)")
    parser.add_argument("--key-columns", nargs="+", default=["PORTAFOGLIO", "DESTINAZIONE_CONTABILE_LOCAL", "ISIN"],
                        help="Columns to use as composite key (default: PORTAFOGLIO DESTINAZIONE_CONTABILE_LOCAL ISIN)")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Numeric tolerance (default: 1e-6)")
    parser.add_argument("--exclude", nargs="*", default=["idasset", "idExecution_tExecution", "DATA_CARICAMENTO"],
                        help="Columns to exclude from comparison")
    parser.add_argument("--max-rows", type=int, default=50, help="Max rows to show in report")
    parser.add_argument("--show-all-diffs", action="store_true", help="Show all differences for each row")
    parser.add_argument("--json", dest="json_out", default=None, help="Write full report as JSON")
    
    args = parser.parse_args()
    
    file_og = Path(args.file_og).expanduser().resolve()
    file_gen = Path(args.file_gen).expanduser().resolve()
    
    if not file_og.exists():
        raise SystemExit(f"File not found: {file_og}")
    if not file_gen.exists():
        raise SystemExit(f"File not found: {file_gen}")
    
    print(f"Analyzing files:")
    print(f"  OG:  {file_og}")
    print(f"  GEN: {file_gen}")
    print(f"  Key columns: {' + '.join(args.key_columns)}")
    print(f"  Excluded columns: {', '.join(args.exclude)}")
    print()
    
    comparisons, stats = analyze_files(
        file_og,
        file_gen,
        delimiter=args.delimiter,
        encoding=args.encoding,
        key_columns=args.key_columns,
        numeric_tolerance=args.tolerance,
        exclude_columns=args.exclude
    )
    
    print_human_report(
        comparisons,
        stats,
        max_rows_to_show=args.max_rows,
        show_all_differences=args.show_all_diffs
    )
    
    # Write JSON report if requested
    if args.json_out:
        json_path = Path(args.json_out).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        report = {
            "summary": asdict(stats),
            "comparisons": []
        }
        
        for comp in comparisons:
            comp_dict = {
                "key": comp.key,
                "idasset_og": comp.idasset_og,
                "idasset_gen": comp.idasset_gen,
                "status": comp.status,
                "differences": [asdict(d) for d in comp.differences]
            }
            report["comparisons"].append(comp_dict)
        
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\nJSON report written to: {json_path}")
    
    print("\n" + "=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
