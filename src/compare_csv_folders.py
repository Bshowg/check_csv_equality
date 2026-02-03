#!/usr/bin/env python3
"""
compare_csv_folders.py

Compare CSV files in two folders (by filename) and report differences:
- file missing/present
- headers differences (including order)
- number of rows
- number of columns
- optional content differences (row-wise, after normalization)

Examples:
  python compare_csv_folders.py /path/A /path/B
  python compare_csv_folders.py /path/A /path/B --recursive --json report.json
  python compare_csv_folders.py /path/A /path/B --delimiter ";" --encoding "utf-8-sig"
  python compare_csv_folders.py /path/A /path/B --content --max-diff-rows 20
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class CsvDiffReport:
    filename: str
    exists_in_a: bool
    exists_in_b: bool

    # Structural
    rows_a: Optional[int] = None
    rows_b: Optional[int] = None
    cols_a: Optional[int] = None
    cols_b: Optional[int] = None

    headers_a: Optional[List[str]] = None
    headers_b: Optional[List[str]] = None

    header_equal: Optional[bool] = None
    header_order_equal: Optional[bool] = None
    header_only_in_a: Optional[List[str]] = None
    header_only_in_b: Optional[List[str]] = None

    # Content (optional)
    content_equal: Optional[bool] = None
    differing_row_count: Optional[int] = None
    sample_differences: Optional[List[Dict]] = None

    error: Optional[str] = None


def list_csv_files(folder: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted([p for p in folder.rglob("*.csv") if p.is_file()])
    return sorted([p for p in folder.glob("*.csv") if p.is_file()])


def key_for_path(folder: Path, path: Path, use_relative: bool) -> str:
    # Matching key: filename only (default) or relative path inside folder
    return str(path.relative_to(folder)) if use_relative else path.name


def build_index(folder: Path, recursive: bool, use_relative: bool, case_insensitive: bool) -> Dict[str, Path]:
    files = list_csv_files(folder, recursive=recursive)
    idx: Dict[str, Path] = {}
    for p in files:
        k = key_for_path(folder, p, use_relative=use_relative)
        if case_insensitive:
            k = k.lower()
        # If duplicates exist, keep first but you could also raise.
        idx.setdefault(k, p)
    return idx


def normalize_cell(s: str, trim: bool, lower: bool) -> str:
    if s is None:
        s = ""
    if trim:
        s = s.strip()
    if lower:
        s = s.lower()
    return s


def cells_equal_with_delta(val_a: str, val_b: str, trim: bool, lower: bool, numeric_delta: float = 0.0) -> bool:
    """
    Compare two cell values with optional numeric delta threshold.
    Returns True if values are equal (considering delta for numeric values).
    """
    # Normalize both values
    norm_a = normalize_cell(val_a, trim, lower)
    norm_b = normalize_cell(val_b, trim, lower)
    
    # If strings are equal, return True
    if norm_a == norm_b:
        return True
    
    # If numeric_delta is specified, try numeric comparison
    if numeric_delta > 0:
        try:
            num_a = float(norm_a)
            num_b = float(norm_b)
            return abs(num_a - num_b) <= numeric_delta
        except (ValueError, OverflowError):
            # Not numeric, fall back to string comparison
            pass
    
    return False


def read_csv_basic(
    path: Path,
    delimiter: str,
    encoding: str,
    trim_cells: bool,
    lower_cells: bool,
) -> Tuple[List[str], int, int]:
    """
    Returns: (headers, n_rows, n_cols)
    n_rows counts data rows (excluding header row).
    n_cols is len(headers).
    """
    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        try:
            headers = next(reader)
        except StopIteration:
            # Empty file
            return [], 0, 0

        headers = [normalize_cell(h, trim=trim_cells, lower=False) for h in headers]
        n_cols = len(headers)

        n_rows = 0
        for _ in reader:
            n_rows += 1

    return headers, n_rows, n_cols


def stable_row_hash(
    row: List[str],
    trim_cells: bool,
    lower_cells: bool,
    numeric_delta: float = 0.0,
) -> str:
    """
    Create a stable hash for row comparison.
    If numeric_delta > 0, rounds numeric values to create comparable hashes.
    """
    normed = []
    for c in row:
        val = normalize_cell(c, trim=trim_cells, lower=lower_cells)
        # If numeric delta is specified, try to normalize numeric values
        if numeric_delta > 0 and val:
            try:
                num_val = float(val)
                # Round to appropriate decimal places based on delta
                decimal_places = max(0, -int(str(numeric_delta).split('.')[-1].rstrip('0').__len__()) + 1)
                val = f"{num_val:.{decimal_places}f}"
            except (ValueError, OverflowError):
                pass
        normed.append(val)
    joined = "\x1f".join(normed)  # unlikely separator
    return hashlib.sha256(joined.encode("utf-8-sig")).hexdigest()


def sort_rows_progressively(
    rows: List[List[str]],
    num_cols: int,
    trim_cells: bool,
    lower_cells: bool,
) -> List[Tuple[int, List[str]]]:
    """
    Sort rows using progressive column ordering.
    First by column 0, then by column 1 if column 0 is equal, etc.
    
    Returns: List of tuples (original_index, row) sorted by row content.
    """
    if not rows or num_cols == 0:
        return [(i, row) for i, row in enumerate(rows)]
    
    # Create tuples of (original_index, row)
    indexed_rows = [(i, row) for i, row in enumerate(rows)]
    
    def sort_key(indexed_row: Tuple[int, List[str]]) -> Tuple:
        row = indexed_row[1]
        # Normalize cells for comparison and create a tuple for sorting
        # Extend row with empty strings if it's shorter than num_cols
        extended_row = row + [''] * (num_cols - len(row))
        return tuple(
            normalize_cell(extended_row[i], trim=trim_cells, lower=lower_cells)
            for i in range(num_cols)
        )
    
    return sorted(indexed_rows, key=sort_key)

 
def compare_csv_content(
    path_a: Path,
    path_b: Path,
    delimiter: str,
    encoding: str,
    trim_cells: bool,
    lower_cells: bool,
    max_diff_rows: int,
    reorder_columns: bool = False,
    numeric_delta: float = 0.0,
) -> Tuple[bool, int, List[Dict]]:
    """
    Compares CSV content checking if all rows in A are present in B.
    Reports count of differing/missing rows and provides samples.
    
    Important: Files are considered EQUAL if all rows in A exist in B.
    Extra rows in B that don't exist in A are NOT counted as differences.
    This implements a subset check: A ⊆ B

    Strategy:
      - read both files completely
      - sort rows by progressive column ordering (col 0, then col 1, etc.)
      - check if each row in A exists in B (with numeric delta tolerance)
      - report rows from A that are missing or different in B
      - extra rows in B are ignored
    """
    diffs: List[Dict] = []
    differing = 0

    # Read all rows from both files
    rows_a = []
    rows_b = []
    headers_a = []
    headers_b = []
    
    with path_a.open("r", encoding=encoding, newline="") as fa:
        ra = csv.reader(fa, delimiter=delimiter)
        try:
            headers_a = next(ra)
        except StopIteration:
            pass
        rows_a = list(ra)
    
    with path_b.open("r", encoding=encoding, newline="") as fb:
        rb = csv.reader(fb, delimiter=delimiter)
        try:
            headers_b = next(rb)
        except StopIteration:
            pass
        rows_b = list(rb)
    
    # Reorder columns in B to match A if requested and headers are equal (same set)
    if reorder_columns and headers_a and headers_b:
        set_a = set(headers_a)
        set_b = set(headers_b)
        if set_a == set_b and headers_a != headers_b:
            # Create mapping from B column positions to A column positions
            reorder_map = [headers_b.index(col) for col in headers_a]
            # Reorder rows_b according to the mapping
            reordered_rows_b = []
            for row in rows_b:
                reordered_row = [''] * len(headers_a)
                for new_idx, old_idx in enumerate(reorder_map):
                    if old_idx < len(row):
                        reordered_row[new_idx] = row[old_idx]
                reordered_rows_b.append(reordered_row)
            rows_b = reordered_rows_b
            headers_b = headers_a.copy()  # Update headers_b to match order
    
    # Determine number of columns for sorting (use max of both)
    num_cols = max(len(headers_a), len(headers_b)) if (headers_a or headers_b) else 0
    
    # Sort both sets of rows using progressive column ordering
    # Returns list of (original_index, row) tuples
    sorted_rows_a = sort_rows_progressively(rows_a, num_cols, trim_cells, lower_cells)
    sorted_rows_b = sort_rows_progressively(rows_b, num_cols, trim_cells, lower_cells)
    
    # Build a hash map of rows in B for efficient lookup
    rows_b_dict = {}
    for orig_idx_b, row_b in sorted_rows_b:
        row_hash = stable_row_hash(row_b, trim_cells, lower_cells, numeric_delta)
        if row_hash not in rows_b_dict:
            rows_b_dict[row_hash] = []
        rows_b_dict[row_hash].append((orig_idx_b, row_b))
    
    # Track diffs by type for max_diff_rows limit per type
    diffs_with_closest_match = []
    diffs_not_found = []
    
    # Check if each row in A exists in B
    for orig_idx_a, row_a in sorted_rows_a:
        row_hash_a = stable_row_hash(row_a, trim_cells, lower_cells, numeric_delta)
        
        if row_hash_a in rows_b_dict:
            # Row exists in B with matching hash - do detailed comparison
            matches = rows_b_dict[row_hash_a]
            found_match = False
            
            for orig_idx_b, row_b in matches:
                # Check if all fields match (with delta for numeric values)
                all_match = True
                max_len = max(len(row_a), len(row_b))
                
                for col_idx in range(max_len):
                    val_a = row_a[col_idx] if col_idx < len(row_a) else ""
                    val_b = row_b[col_idx] if col_idx < len(row_b) else ""
                    
                    if not cells_equal_with_delta(val_a, val_b, trim_cells, lower_cells, numeric_delta):
                        all_match = False
                        break
                
                if all_match:
                    found_match = True
                    # Remove this match from dict so it's not matched again
                    rows_b_dict[row_hash_a].remove((orig_idx_b, row_b))
                    if not rows_b_dict[row_hash_a]:
                        del rows_b_dict[row_hash_a]
                    break
            
            if not found_match:
                # Hash matched but detailed comparison failed - shouldn't happen with good hash
                differing += 1
                if len(diffs_not_found) < max_diff_rows:
                    diffs_not_found.append({
                        "row_index_a_1_based": orig_idx_a + 2,
                        "status": "not_found_in_b",
                        "row": row_a
                    })
        else:
            # Row from A not found in B
            differing += 1
            
            # Try to find the closest match in B for reporting
            closest_match = None
            min_diff_count = float('inf')
            
            for b_rows in rows_b_dict.values():
                for orig_idx_b, row_b in b_rows:
                    diff_count = 0
                    differing_fields = []
                    max_len = max(len(row_a), len(row_b))
                    
                    for col_idx in range(max_len):
                        val_a = row_a[col_idx] if col_idx < len(row_a) else ""
                        val_b = row_b[col_idx] if col_idx < len(row_b) else ""
                        
                        if not cells_equal_with_delta(val_a, val_b, trim_cells, lower_cells, numeric_delta):
                            diff_count += 1
                            field_info = {
                                "column_index": col_idx,
                                "a": val_a,
                                "b": val_b,
                            }
                            # Add column name if available
                            if col_idx < len(headers_a):
                                field_info["column_name"] = headers_a[col_idx]
                            elif col_idx < len(headers_b):
                                field_info["column_name"] = headers_b[col_idx]
                            differing_fields.append(field_info)
                    
                    if diff_count < min_diff_count:
                        min_diff_count = diff_count
                        closest_match = (orig_idx_b, differing_fields)
            
            if closest_match and len(diffs_with_closest_match) < max_diff_rows:
                orig_idx_b, differing_fields = closest_match
                diffs_with_closest_match.append({
                    "row_index_a_1_based": orig_idx_a + 2,
                    "row_index_b_1_based": orig_idx_b + 2,
                    "status": "not_found_in_b_closest_match",
                    "differing_fields": differing_fields,
                })
            elif not closest_match and len(diffs_not_found) < max_diff_rows:
                diffs_not_found.append({
                    "row_index_a_1_based": orig_idx_a + 2,
                    "status": "not_found_in_b",
                    "row": row_a
                })
    
    # Combine diffs from both categories
    diffs = diffs_with_closest_match + diffs_not_found

    return differing == 0, differing, diffs


def header_analysis(headers_a: List[str], headers_b: List[str]) -> Tuple[bool, bool, List[str], List[str]]:
    set_a = set(headers_a)
    set_b = set(headers_b)
    only_a = sorted(list(set_a - set_b))
    only_b = sorted(list(set_b - set_a))
    equal_set = (set_a == set_b)
    equal_order = (headers_a == headers_b)
    return equal_set, equal_order, only_a, only_b


def compare_one(
    filename_key: str,
    path_a: Optional[Path],
    path_b: Optional[Path],
    delimiter: str,
    encoding: str,
    trim_cells: bool,
    lower_cells: bool,
    do_content: bool,
    max_diff_rows: int,
    reorder_columns: bool = False,
    numeric_delta: float = 0.0,
) -> CsvDiffReport:
    rep = CsvDiffReport(
        filename=filename_key,
        exists_in_a=path_a is not None,
        exists_in_b=path_b is not None,
    )

    if path_a is None or path_b is None:
        return rep

    try:
        ha, ra, ca = read_csv_basic(path_a, delimiter, encoding, trim_cells, lower_cells)
        hb, rb, cb = read_csv_basic(path_b, delimiter, encoding, trim_cells, lower_cells)
        rep.rows_a = ra
        rep.rows_b = rb
        rep.cols_a = ca
        rep.cols_b = cb

        equal_set, equal_order, only_a, only_b = header_analysis(ha, hb)
        rep.header_equal = equal_set
        rep.header_order_equal = equal_order
        rep.header_only_in_a = only_a
        rep.header_only_in_b = only_b
        
        # Only include headers in report if they differ
        if not equal_set:
            rep.headers_a = ha
            rep.headers_b = hb

        if do_content:
            eq, differing, samples = compare_csv_content(
                path_a, path_b, delimiter, encoding, trim_cells, lower_cells, max_diff_rows, reorder_columns, numeric_delta
            )
            rep.content_equal = eq
            rep.differing_row_count = differing
            rep.sample_differences = samples

    except Exception as e:
        rep.error = f"{type(e).__name__}: {e}"

    return rep


def print_human_report(reports: List[CsvDiffReport]) -> None:
    for r in reports:
        print("=" * 80)
        print(f"{r.filename}")

        if not r.exists_in_a:
            print("  - Missing in folder A")
            continue
        if not r.exists_in_b:
            print("  - Missing in folder B")
            continue

        if r.error:
            print(f"  - ERROR reading/comparing: {r.error}")
            continue

        print(f"  - Rows:    A={r.rows_a}  B={r.rows_b}")
        print(f"  - Columns: A={r.cols_a}  B={r.cols_b}")

        if r.header_equal is False:
            print("  - Headers differ (set mismatch)")
            if r.header_only_in_a:
                print(f"    * Only in A: {r.header_only_in_a}")
            if r.header_only_in_b:
                print(f"    * Only in B: {r.header_only_in_b}")
        else:
            # Same set
            if r.header_order_equal:
                print("  - Headers: same (including order)")
            else:
                print("  - Headers: same set, different order")

        if r.content_equal is not None:
            if r.content_equal:
                print("  - Content: identical (row-by-row)")
            else:
                print(f"  - Content: differs in {r.differing_row_count} row position(s)")
                if r.sample_differences:
                    print("  - Sample differences:")
                    for d in r.sample_differences:
                        if "differing_fields" in d:
                            idx_a = d.get("row_index_a_1_based")
                            status = d.get("status", "")
                            if status == "not_found_in_b_closest_match":
                                idx_b = d.get("row_index_b_1_based")
                                print(f"    * Row A:{idx_a} NOT FOUND in B (closest match B:{idx_b}) - {len(d['differing_fields'])} field(s) differ:")
                            else:
                                idx_b = d.get("row_index_b_1_based")
                                print(f"    * Row A:{idx_a} vs B:{idx_b} - {len(d['differing_fields'])} field(s) differ:")
                            for field in d["differing_fields"]:
                                col_name = field.get("column_name", f"col_{field['column_index']}")
                                print(f"      - {col_name}: A='{field['a']}' vs B='{field['b']}'")
                        elif "status" in d:
                            status = d["status"]
                            if status == "not_found_in_b":
                                idx = d.get("row_index_a_1_based")
                                print(f"    * Row A:{idx}: NOT FOUND in B")
                                print(f"      {d.get('row', [])}")
                            elif status == "only_in_a":
                                idx = d.get("row_index_a_1_based")
                                print(f"    * Row A:{idx}: only_in_a")
                                print(f"      {d.get('row', [])}")
                            elif status == "only_in_b":
                                idx = d.get("row_index_b_1_based")
                                print(f"    * Row B:{idx}: only_in_b")
                                print(f"      {d.get('row', [])}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder_a", type=str, help="First folder of CSV files")
    ap.add_argument("folder_b", type=str, help="Second folder of CSV files")
    ap.add_argument("--recursive", action="store_true", help="Search subfolders too")
    ap.add_argument("--match-relative-paths", action="store_true", default=True,
                    help="Match files by relative path (default: True). Use --no-match-relative-paths to match by filename only.")
    ap.add_argument("--no-match-relative-paths", action="store_false", dest="match_relative_paths",
                    help="Match files by filename only instead of relative path")
    ap.add_argument("--case-insensitive", action="store_true", help="Case-insensitive filename matching")
    ap.add_argument("--delimiter", default=";", help="CSV delimiter (default: ,)")
    ap.add_argument("--encoding", default="utf-8-sig", help="File encoding (default: utf-8)")
    ap.add_argument("--trim-cells", action="store_true", help="Trim whitespace in cells during content compare")
    ap.add_argument("--lower-cells", action="store_true", help="Lowercase cells during content compare")
    ap.add_argument("--content", action="store_true", help="Also compare content row-by-row (can be slower)")
    ap.add_argument("--max-diff-rows", type=int, default=1, help="Max differing rows to print/save per type (default: 1)")
    ap.add_argument("--reorder-columns", action="store_true", help="Reorder columns in B to match A if headers are equal (same set)")
    ap.add_argument("--numeric-delta", type=float, default=0.0, help="Delta threshold for numeric value comparison (default: 0.0 - exact match)")
    ap.add_argument("--json", dest="json_out", default=None, help="Write full report as JSON to this path")

    args = ap.parse_args()

    folder_a = Path(args.folder_a).expanduser().resolve()
    folder_b = Path(args.folder_b).expanduser().resolve()

    if not folder_a.exists() or not folder_a.is_dir():
        raise SystemExit(f"Folder A does not exist or is not a directory: {folder_a}")
    if not folder_b.exists() or not folder_b.is_dir():
        raise SystemExit(f"Folder B does not exist or is not a directory: {folder_b}")

    idx_a = build_index(folder_a, recursive=args.recursive, use_relative=args.match_relative_paths,
                        case_insensitive=args.case_insensitive)
    idx_b = build_index(folder_b, recursive=args.recursive, use_relative=args.match_relative_paths,
                        case_insensitive=args.case_insensitive)

    # Compare: "for each file in folder A"
    reports: List[CsvDiffReport] = []
    for key, path_a in idx_a.items():
        path_b = idx_b.get(key)
        reports.append(
            compare_one(
                filename_key=key,
                path_a=path_a,
                path_b=path_b,
                delimiter=args.delimiter,
                encoding=args.encoding,
                trim_cells=args.trim_cells,
                lower_cells=args.lower_cells,
                do_content=args.content,
                max_diff_rows=args.max_diff_rows,
                reorder_columns=args.reorder_columns,
                numeric_delta=args.numeric_delta,
            )
        )

    # Print human report
    print_human_report(reports)

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        diff_reports=[asdict(r) for r in reports if r.content_equal is False]
        # Create summary with general info
        summary = {
            "folder_a": str(folder_a),
            "folder_b": str(folder_b),
            "total_files_in_a": len(idx_a),
            "total_files_in_b": len(idx_b),
            "files_compared": len(reports),
            "files_only_in_a": sum(1 for r in reports if r.exists_in_a and not r.exists_in_b),
            "files_only_in_b": sum(1 for r in reports if not r.exists_in_a and r.exists_in_b),
            "files_in_both": sum(1 for r in reports if r.exists_in_a and r.exists_in_b),
            "files_with_differences": sum(1 for r in diff_reports),
            "options": {
                "recursive": args.recursive,
                "match_relative_paths": args.match_relative_paths,
                "case_insensitive": args.case_insensitive,
                "delimiter": args.delimiter,
                "encoding": args.encoding,
                "trim_cells": args.trim_cells,
                "lower_cells": args.lower_cells,
                "content_comparison": args.content,
                "reorder_columns": args.reorder_columns,
                "numeric_delta": args.numeric_delta,
            }
        }
        
        json_output = {
            "summary": summary,
            "reports": diff_reports
        }
        
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)
        print("=" * 80)
        print(f"Wrote JSON report to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
