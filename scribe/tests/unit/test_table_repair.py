from unittest.mock import MagicMock

from scribe.tools.ocr_processor import (
    _build_markdown_table,
    _count_data_columns,
    _has_broken_table,
    _replace_broken_table,
)


class TestHasBrokenTable:
    def test_detects_br_heavy_cells(self):
        brs = "<br>".join([str(i) for i in range(12)])
        md = f"| Label | {brs} |\n| --- | --- |\n| Row | data |"
        assert _has_broken_table(md) is True

    def test_clean_table(self):
        md = "| A | B | C |\n| --- | --- | --- |\n| 1 | 2 | 3 |"
        assert _has_broken_table(md) is False

    def test_no_table(self):
        md = "This is plain text\nwith no table at all."
        assert _has_broken_table(md) is False

    def test_few_br_tags(self):
        md = "| Label | a<br>b<br>c |\n| --- | --- |"
        assert _has_broken_table(md) is False


class TestCountDataColumns:
    def test_standard_bwa(self):
        """12 columns: 4 label fragments + 8 numeric data columns."""
        rows = [
            [
                "Konto",
                "",
                "",
                "Bezeichnung",
                "1.000,00",
                "2.000,00",
                "3.000,00",
                "4.000,00",
                "5.000,00",
                "6.000,00",
                "7.000,00",
                "8.000,00",
            ],
            [
                "4000",
                "",
                "",
                "Umsatzerlöse",
                "10.000,00",
                "20.000,00",
                "30.000,00",
                "40.000,00",
                "50.000,00",
                "60.000,00",
                "70.000,00",
                "80.000,00",
            ],
            [
                "4100",
                "",
                "",
                "Sonstige",
                "1.000,00",
                "2.000,00",
                "3.000,00",
                "4.000,00",
                "5.000,00",
                "6.000,00",
                "7.000,00",
                "8.000,00",
            ],
        ]
        assert _count_data_columns(rows) == 8

    def test_empty_rows(self):
        assert _count_data_columns([]) == 8

    def test_no_cells(self):
        assert _count_data_columns([[]]) == 8


class TestBuildMarkdownTable:
    def test_merges_labels(self):
        rows = [
            ["Konto", "", "Bezeichnung", "Jan", "Feb"],
            ["4000", "", "Umsatzerlöse", "10.000", "20.000"],
        ]
        result = _build_markdown_table(rows, data_cols=2)
        lines = result.splitlines()
        assert len(lines) == 3  # header + separator + data
        assert "Konto Bezeichnung" in lines[0]
        assert "Jan" in lines[0]
        assert "---" in lines[1]
        assert "4000 Umsatzerlöse" in lines[2]
        assert "10.000" in lines[2]

    def test_handles_none_cells(self):
        rows = [
            ["Header", None, "100"],
            [None, None, "200"],
        ]
        result = _build_markdown_table(rows, data_cols=1)
        lines = result.splitlines()
        assert "Header" in lines[0]
        assert "100" in lines[0]
        assert "200" in lines[2]

    def test_skips_empty_rows(self):
        rows = [
            ["Header", "Data"],
            ["", ""],
            ["Row", "123"],
        ]
        result = _build_markdown_table(rows, data_cols=1)
        lines = result.splitlines()
        # header + separator + 1 data row = 3 lines (empty row skipped)
        assert len(lines) == 3

    def test_empty_rows_list(self):
        assert _build_markdown_table([], data_cols=2) == ""


class TestReplaceBrokenTable:
    def test_preserves_non_table_content(self):
        page_md = "Header text\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\nFooter text"

        mock_table = MagicMock()
        mock_table.extract.return_value = [["Label", "Value"], ["X", "100"]]

        mock_tables = MagicMock()
        mock_tables.tables = [mock_table]
        mock_tables.__getitem__ = lambda self, idx: mock_table

        mock_page = MagicMock()
        mock_page.find_tables.return_value = mock_tables

        result = _replace_broken_table(page_md, mock_page)
        assert "Header text" in result
        assert "Footer text" in result

    def test_no_tables_returns_empty(self):
        mock_tables = MagicMock()
        mock_tables.tables = []
        mock_page = MagicMock()
        mock_page.find_tables.return_value = mock_tables

        result = _replace_broken_table("| A | B |", mock_page)
        assert result == ""

    def test_no_table_in_md_returns_empty(self):
        mock_table = MagicMock()
        mock_table.extract.return_value = [["A", "B"]]
        mock_tables = MagicMock()
        mock_tables.tables = [mock_table]
        mock_tables.__getitem__ = lambda self, idx: mock_table

        mock_page = MagicMock()
        mock_page.find_tables.return_value = mock_tables

        result = _replace_broken_table("No table here", mock_page)
        assert result == ""
