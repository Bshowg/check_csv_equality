# Check CSV Equality

A powerful Python tool for comparing CSV files in two folders with advanced features including subset checking, numeric tolerance, and intelligent row matching. Perfect for validating data migrations, testing data transformations, or comparing datasets.

## Features

### Core Comparison
- **Subset Checking**: Verify all rows in folder A exist in folder B (A ⊆ B)
- **Intelligent Row Matching**: Rows are sorted and matched regardless of original order
- **Numeric Tolerance**: Compare numeric values with configurable delta threshold
- **Column Reordering**: Automatically align columns when headers match but are in different order
- **Progressive Sorting**: Multi-column sort (col 0, then col 1, etc.) for consistent comparison

### Structural Analysis
- Detect missing files in either folder
- Compare headers (names and order)
- Compare row and column counts
- Match files by name or relative path

### Flexible Options
- Recursive folder scanning
- Configurable delimiter and encoding
- Cell normalization (trim whitespace, lowercase)
- Per-type difference limits (closest match vs. not found)

### Output Formats
- **Console**: Human-readable formatted output
- **JSON**: Structured report with detailed differences
- **HTML**: Beautiful visual report with side-by-side comparisons

## Project Structure

```
check_csv_equality/
├── src/
│   ├── __init__.py
│   ├── compare_csv_folders.py   # Main comparison engine
│   ├── visualize_report.py      # Report visualization tool
│   └── analyze_asset_portafoglio.py
├── tests/
│   ├── __init__.py
│   └── test_compare_csv_folders.py
├── data/                         # Ignored by git
│   ├── og/                       # Original data
│   └── gen/                      # Generated/compared data
├── .github/
│   └── copilot-instructions.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Comparison

Simple structural comparison (headers, row counts):
```bash
python -m src.compare_csv_folders folder_a folder_b
```

### Content Comparison with Subset Check

Check if all rows in A exist in B (ignores extra rows in B):
```bash
python -m src.compare_csv_folders folder_a folder_b --content
```

### Recommended: Full Comparison with All Features

```bash
python -m src.compare_csv_folders data/og data/gen --content --recursive --reorder-columns  --numeric-delta 0.0001 --delimiter ";" --encoding "utf-8-sig" --json report.json
```

This command:
- Compares all CSV files recursively
- Checks if rows in A exist in B
- Reorders columns in B to match A
- Treats numeric values as equal within 0.0001 delta
- Uses semicolon delimiter and UTF-8 with BOM encoding
- Generates JSON report

### Visualize Results

**Console output:**
```bash
python -m src.visualize_report report.json
```

**HTML report:**
```bash
python -m src.visualize_report report.json --format html --output report.html
```

### Advanced Examples

**Custom delimiter and encoding:**
```bash
python -m src.compare_csv_folders folder_a folder_b \
  --delimiter ";" \
  --encoding "utf-8-sig"
```

**With cell normalization:**
```bash
python -m src.compare_csv_folders folder_a folder_b \
  --content \
  --trim-cells \
  --lower-cells
```

**Match by filename only (not relative path):**
```bash
python -m src.compare_csv_folders folder_a folder_b \
  --recursive \
  --no-match-relative-paths
```
### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--recursive` | Search subfolders | False |
| `--match-relative-paths` | Match files by relative path (default on) | True |
| `--no-match-relative-paths` | Match files by filename only | - |
| `--case-insensitive` | Case-insensitive filename matching | False |
| `--delimiter` | CSV delimiter | `,` |
| `--encoding` | File encoding | `utf-8` |
| `--trim-cells` | Trim whitespace in cells | False |
| `--lower-cells` | Lowercase cells during comparison | False |
| `--content` | Compare content row-by-row | False |
| `--reorder-columns` | Reorder columns in B to match A if headers match | False |
| `--numeric-delta` | Delta threshold for numeric comparison | `0.0` (exact) |
| `--max-diff-rows` | Max differing rows to display per type | `1` |
| `--json` | Output JSON report to specified path | None |

## How It Works

### Subset Checking (A ⊆ B)

The tool checks if **all rows in folder A exist in folder B**. This is perfect for:
- Validating that reference data (A) is present in production data (B)
- Checking if expected results (A) appear in actual results (B)
- Ensuring data hasn't been lost during migration

**Important**: Extra rows in B that don't exist in A are **not** considered errors. Files are equal if A ⊆ B.

### Row Matching Algorithm

1. **Sort rows progressively** by all columns (col 0, then col 1, etc.)
2. **Build hash map** of rows in B for efficient lookup
3. **For each row in A**, search for matching row in B
4. **Numeric tolerance**: If `--numeric-delta` is set, numeric values within delta are considered equal
5. **Report differences**: Shows closest match if exact match not found

### Field-Level Reporting

When rows differ, only the **specific fields** that are different are reported:
- Column name (if available)
- Value in file A
- Value in file B
- Original row indices in both files

## Output Formats

### JSON Report Structure

```json
{
  "summary": {
    "folder_a": "path/to/data/og",
    "folder_b": "path/to/data/gen",
    "total_files_in_a": 10,
    "total_files_in_b": 10,
    "files_with_differences": 2,
    "options": { ... }
  },
  "reports": [
    {
      "filename": "file.csv",
      "content_equal": false,
      "differing_row_count": 3,
      "sample_differences": [
        {
          "row_index_a_1_based": 5,
          "row_index_b_1_based": 7,
          "status": "not_found_in_b_closest_match",
          "differing_fields": [
            {
              "column_name": "price",
              "column_index": 2,
              "a": "10.5",
              "b": "10.4999"
            }
          ]
        }
      ]
    }
  ]
}
```

### Console Output

```
================================================================================
CSV COMPARISON REPORT
================================================================================

SUMMARY
--------------------------------------------------------------------------------
Files with differences: 2

[1] data/file.csv
--------------------------------------------------------------------------------
  Rows:    A=100  B=105
  Columns: A=5    B=5
  ✓ Headers: same (including order)
  ❌ Content: 3 row(s) differ

  Sample Differences:
    * Row A:5 vs B:7 - 1 field(s) differ:
      - price: A='10.5' vs B='10.4999'
```

### HTML Report

Beautiful, responsive web page with:
- Color-coded statistics
- Side-by-side value comparisons
- Filterable differences
- Professional gradient design

## Testing

Run tests with:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=src tests/
```

## Use Cases

### Data Migration Validation
```bash
# Verify all original records exist in migrated database
python -m src.compare_csv_folders original_export/ migrated_export/ \
  --content --recursive --numeric-delta 0.001
```

### ETL Testing
```bash
# Check if transformed data matches expected output
python -m src.compare_csv_folders expected/ actual/ \
  --content --reorder-columns --trim-cells
```

### Data Synchronization
```bash
# Verify source data is present in target system
python -m src.compare_csv_folders source/ target/ \
  --content --recursive --json sync_report.json
```

### Financial Data Comparison
```bash
# Compare financial reports with penny-level precision
python -m src.compare_csv_folders q1_report/ q2_report/ \
  --content --numeric-delta 0.01 --delimiter ";" --encoding "utf-8-sig"
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Development

- Source code: `src/`
- Tests: `tests/`
- Python 3.8+ required




