"""Tests for compare_csv_folders module."""

import pytest
from pathlib import Path
from src.compare_csv_folders import (
    normalize_cell,
    header_analysis,
    stable_row_hash,
)


class TestNormalizeCell:
    def test_basic_string(self):
        assert normalize_cell("test", trim=False, lower=False) == "test"
    
    def test_trim_whitespace(self):
        assert normalize_cell("  test  ", trim=True, lower=False) == "test"
    
    def test_lowercase(self):
        assert normalize_cell("TEST", trim=False, lower=True) == "test"
    
    def test_trim_and_lowercase(self):
        assert normalize_cell("  TEST  ", trim=True, lower=True) == "test"


class TestHeaderAnalysis:
    def test_identical_headers(self):
        headers_a = ["id", "name", "value"]
        headers_b = ["id", "name", "value"]
        equal_set, equal_order, only_a, only_b = header_analysis(headers_a, headers_b)
        assert equal_set is True
        assert equal_order is True
        assert only_a == []
        assert only_b == []
    
    def test_different_order(self):
        headers_a = ["id", "name", "value"]
        headers_b = ["name", "id", "value"]
        equal_set, equal_order, only_a, only_b = header_analysis(headers_a, headers_b)
        assert equal_set is True
        assert equal_order is False
        assert only_a == []
        assert only_b == []
    
    def test_different_headers(self):
        headers_a = ["id", "name", "value"]
        headers_b = ["id", "age", "value"]
        equal_set, equal_order, only_a, only_b = header_analysis(headers_a, headers_b)
        assert equal_set is False
        assert only_a == ["name"]
        assert only_b == ["age"]


class TestStableRowHash:
    def test_identical_rows(self):
        row1 = ["a", "b", "c"]
        row2 = ["a", "b", "c"]
        h1 = stable_row_hash(row1, trim=False, lower=False)
        h2 = stable_row_hash(row2, trim=False, lower=False)
        assert h1 == h2
    
    def test_different_rows(self):
        row1 = ["a", "b", "c"]
        row2 = ["a", "b", "d"]
        h1 = stable_row_hash(row1, trim=False, lower=False)
        h2 = stable_row_hash(row2, trim=False, lower=False)
        assert h1 != h2
    
    def test_trim_affects_hash(self):
        row1 = ["a", "b ", "c"]
        row2 = ["a", "b", "c"]
        h1 = stable_row_hash(row1, trim=True, lower=False)
        h2 = stable_row_hash(row2, trim=True, lower=False)
        assert h1 == h2
