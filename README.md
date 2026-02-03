# Check CSV Equality

A Python tool for comparing CSV files in two folders and reporting differences including structural changes, header differences, and optional content comparison.

## Features

- Compare CSV files by filename or relative path
- Detect missing files in either folder
- Compare headers (names and order)
- Compare row and column counts
- Optional row-by-row content comparison
- Flexible options for normalization (trim, lowercase)
- JSON report output
- Recursive folder scanning

## Project Structure

```
check_csv_equality/
├── src/
│   ├── __init__.py
│   └── compare_csv_folders.py
├── tests/
│   ├── __init__.py
│   └── test_compare_csv_folders.py
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

Basic comparison:
```bash
python -m src.compare_csv_folders folder_a folder_b
```

Compare with content checking:
```bash
python -m src.compare_csv_folders folder_a folder_b --content
```

Recursive comparison with JSON output:
```bash
python -m src.compare_csv_folders data/og data/gen --content --recursive --json report.json --delimiter ";" --encoding "utf-8-sig"
```

Custom delimiter and encoding:
```bash
python -m src.compare_csv_folders folder_a folder_b  --delimiter ";" --encoding "utf-8-sig"
```
Full 
Recursive comparison with JSON output:
```bash
python -m src.compare_csv_folders data/og data/gen --recursive --json report.json --delimiter ";" --encoding "utf-8-sig"
```
### Options

- `--recursive`: Search subfolders
- `--match-relative-paths`: Match files by relative path (useful with --recursive)
- `--case-insensitive`: Case-insensitive filename matching
- `--delimiter`: CSV delimiter (default: ,)
- `--encoding`: File encoding (default: utf-8)
- `--trim-cells`: Trim whitespace in cells during comparison
- `--lower-cells`: Lowercase cells during comparison
- `--content`: Compare content row-by-row (can be slower)
- `--max-diff-rows`: Max differing rows to display (default: 10)
- `--json`: Write full report as JSON to specified path

## Testing

Run tests with:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=src tests/
```

## Development

- Source code: `src/`
- Tests: `tests/`
- Follow PEP 8 style guidelines
- Write unit tests for all new features

## License

[Add your license here]
