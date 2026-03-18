"""Unit tests for src/exporter.py."""

from __future__ import annotations

import textwrap
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.exporter import ExportResult, GoogleDocsExporter

# ---------------------------------------------------------------------------
# Shared config stub (ExportGoogleDocsConfig does not exist yet)
# ---------------------------------------------------------------------------

_CFG = SimpleNamespace(
    enabled=True,
    credentials_path="credentials.json",
    folder_id="test-folder-id",
    max_retries=3,
)


def _make_exporter(
    cfg: SimpleNamespace | None = None,
) -> GoogleDocsExporter:
    """Create a GoogleDocsExporter with a stub config."""
    return GoogleDocsExporter(cfg or _CFG)


# ===================================================================
# Markdown → HTML conversion
# ===================================================================


class TestMdToHtml:
    """Tests for GoogleDocsExporter._md_to_html."""

    def test_md_to_html_headings(self) -> None:
        exp = _make_exporter()
        html = exp._md_to_html("# Heading 1\n\n## Heading 2")
        assert "<h1>" in html and "Heading 1" in html
        assert "<h2>" in html and "Heading 2" in html

    def test_md_to_html_bold(self) -> None:
        exp = _make_exporter()
        html = exp._md_to_html("This is **bold** text")
        assert "<strong>bold</strong>" in html

    def test_md_to_html_lists(self) -> None:
        exp = _make_exporter()
        md_text = "- item one\n- item two\n- item three"
        html = exp._md_to_html(md_text)
        assert "<ul>" in html
        assert "<li>" in html
        assert "item one" in html
        assert "item two" in html
        assert "item three" in html

    def test_md_to_html_tables(self) -> None:
        exp = _make_exporter()
        md_table = textwrap.dedent("""\
            | Speaker | Duration |
            |---------|----------|
            | Alice   | 10:00    |
            | Bob     | 05:30    |
        """)
        html = exp._md_to_html(md_table)
        assert "<table>" in html
        assert "<th>" in html or "<td>" in html
        assert "Alice" in html
        assert "Bob" in html

    def test_md_to_html_japanese(self) -> None:
        exp = _make_exporter()
        md_text = "# 会議議事録\n\n## 要約\n\nこれはテストです。参加者：田中太郎"
        html = exp._md_to_html(md_text)
        assert "会議議事録" in html
        assert "要約" in html
        assert "これはテストです" in html
        assert "田中太郎" in html

    def test_md_to_html_full_minutes(self) -> None:
        """Realistic end-to-end minutes markdown conversion."""
        exp = _make_exporter()
        md = textwrap.dedent("""\
            # 会議議事録 — 2026-03-17

            ## 参加者
            - Alice
            - Bob

            ## 要約
            プロジェクトの進捗確認を行った。

            ## 議論内容

            ### 1. 進捗報告
            **Alice** がフロントエンド実装の完了を報告した。

            ### 2. 次回アクション

            | 担当者 | タスク | 期限 |
            |--------|--------|------|
            | Alice  | レビュー | 3/20 |
            | Bob    | テスト | 3/22 |

            ## 決定事項
            - リリース日は3月25日に決定
        """)
        html = exp._md_to_html(md)

        # Structural checks
        assert "<h1>" in html
        assert "<h2>" in html
        assert "<h3>" in html
        assert "<ul>" in html
        assert "<li>" in html
        assert "<table>" in html
        assert "<strong>" in html

        # Content checks
        assert "会議議事録" in html
        assert "2026-03-17" in html
        assert "Alice" in html
        assert "Bob" in html
        assert "プロジェクトの進捗確認" in html
        assert "リリース日は3月25日に決定" in html

    def test_md_to_html_wraps_in_html_document(self) -> None:
        exp = _make_exporter()
        html = exp._md_to_html("Hello")
        assert html.startswith("<!DOCTYPE html>")
        assert "<html>" in html
        assert '<meta charset="utf-8">' in html
        assert "<body>" in html
        assert "</body></html>" in html


# ===================================================================
# Export (async, Google Drive API mocked)
# ===================================================================


class _HttpError(Exception):
    """Fake HTTP error with a resp attribute, mimicking googleapiclient errors."""

    def __init__(self, status: int, message: str = "") -> None:
        self.resp = SimpleNamespace(status=status)
        super().__init__(message or f"{status} error")


def _mock_drive_service(doc_id: str = "doc-123", url: str = "https://docs.google.com/doc-123"):
    """Create a mock Google Drive service that returns success."""
    mock_execute = MagicMock(return_value={"id": doc_id, "webViewLink": url})
    mock_create = MagicMock()
    mock_create.return_value.execute = mock_execute
    mock_files = MagicMock()
    mock_files.return_value.create = mock_create
    service = MagicMock()
    service.files = mock_files
    return service


class TestExportSuccess:
    @pytest.mark.asyncio
    async def test_export_success(self) -> None:
        """Successful export returns ExportResult with url and doc_id."""
        exp = _make_exporter()
        exp._service = _mock_drive_service(
            doc_id="abc-123",
            url="https://docs.google.com/document/d/abc-123/edit",
        )

        result = await exp.export("# Test Minutes", "Test Meeting 2026-03-17")

        assert isinstance(result, ExportResult)
        assert result.success is True
        assert result.doc_id == "abc-123"
        assert result.url == "https://docs.google.com/document/d/abc-123/edit"
        assert result.error is None


class TestExportRetry:
    @pytest.mark.asyncio
    async def test_export_retry_on_500(self) -> None:
        """500 errors trigger retry."""
        exp = _make_exporter()

        success_result = {"id": "retry-doc", "webViewLink": "https://docs.google.com/retry-doc"}

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _HttpError(500, "500 Internal Server Error")
            return success_result

        mock_execute = MagicMock(side_effect=side_effect)
        mock_create = MagicMock()
        mock_create.return_value.execute = mock_execute
        mock_files = MagicMock()
        mock_files.return_value.create = mock_create
        service = MagicMock()
        service.files = mock_files
        exp._service = service

        result = await exp.export("# Minutes", "Retry Test")

        assert result.success is True
        assert result.doc_id == "retry-doc"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_export_no_retry_on_400(self) -> None:
        """400 errors should not be retried (non-retryable client error)."""
        exp = _make_exporter()

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise _HttpError(400, "400 Bad Request")

        mock_execute = MagicMock(side_effect=side_effect)
        mock_create = MagicMock()
        mock_create.return_value.execute = mock_execute
        mock_files = MagicMock()
        mock_files.return_value.create = mock_create
        service = MagicMock()
        service.files = mock_files
        exp._service = service

        result = await exp.export("# Minutes", "No Retry Test")

        assert result.success is False
        assert result.error is not None
        # Should have attempted only once, then broken out
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_export_max_retries_exhausted(self) -> None:
        """When all retries are exhausted, ExportResult(success=False) is returned."""
        cfg = SimpleNamespace(
            enabled=True,
            credentials_path="credentials.json",
            folder_id="test-folder-id",
            max_retries=3,
        )
        exp = _make_exporter(cfg)

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ConnectionError("network down")

        mock_execute = MagicMock(side_effect=side_effect)
        mock_create = MagicMock()
        mock_create.return_value.execute = mock_execute
        mock_files = MagicMock()
        mock_files.return_value.create = mock_create
        service = MagicMock()
        service.files = mock_files
        exp._service = service

        result = await exp.export("# Minutes", "Exhaust Test")

        assert result.success is False
        assert "network down" in result.error
        assert call_count == 3


class TestExportNeverRaises:
    @pytest.mark.asyncio
    async def test_export_never_raises(self) -> None:
        """Even on completely unexpected errors, export returns ExportResult."""
        exp = _make_exporter()

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("completely unexpected!")

        mock_execute = MagicMock(side_effect=side_effect)
        mock_create = MagicMock()
        mock_create.return_value.execute = mock_execute
        mock_files = MagicMock()
        mock_files.return_value.create = mock_create
        service = MagicMock()
        service.files = mock_files
        exp._service = service

        # Should NOT raise
        result = await exp.export("# Minutes", "Never Raises Test")

        assert isinstance(result, ExportResult)
        assert result.success is False
        assert result.error is not None
        assert "completely unexpected" in result.error


# ===================================================================
# Service building
# ===================================================================


class TestBuildService:
    def test_build_service_missing_creds(self, tmp_path) -> None:
        """ExportError raised when credentials file does not exist."""
        cfg = SimpleNamespace(
            enabled=True,
            credentials_path=str(tmp_path / "nonexistent-creds.json"),
            folder_id="test-folder-id",
            max_retries=3,
        )
        exp = _make_exporter(cfg)

        from src.errors import ExportError

        with pytest.raises(ExportError, match="Credentials file not found"):
            exp._build_service()

    def test_build_service_caching(self, tmp_path) -> None:
        """Service is built once and cached on subsequent calls."""
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text("{}", encoding="utf-8")

        cfg = SimpleNamespace(
            enabled=True,
            credentials_path=str(creds_file),
            folder_id="test-folder-id",
            max_retries=3,
        )
        exp = _make_exporter(cfg)

        with patch("google.oauth2.service_account.Credentials") as mock_creds_cls, \
             patch("googleapiclient.discovery.build") as mock_build:
            mock_creds_cls.from_service_account_file.return_value = MagicMock()
            mock_build.return_value = MagicMock()

            service1 = exp._build_service()
            service2 = exp._build_service()

            assert service1 is service2
            # Credentials and build should each be called only once
            mock_creds_cls.from_service_account_file.assert_called_once()
            mock_build.assert_called_once()


# ===================================================================
# ExportResult dataclass
# ===================================================================


class TestExportResult:
    def test_success_result(self) -> None:
        r = ExportResult(success=True, url="https://example.com", doc_id="d1")
        assert r.success is True
        assert r.url == "https://example.com"
        assert r.doc_id == "d1"
        assert r.error is None

    def test_failure_result(self) -> None:
        r = ExportResult(success=False, error="something went wrong")
        assert r.success is False
        assert r.url is None
        assert r.doc_id is None
        assert r.error == "something went wrong"

    def test_frozen(self) -> None:
        r = ExportResult(success=True)
        with pytest.raises(AttributeError):
            r.success = False  # type: ignore[misc]
