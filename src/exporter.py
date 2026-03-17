"""Export meeting minutes to Google Docs via Google Drive HTML upload."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import ExportGoogleDocsConfig
from src.errors import ExportError

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/drive.file"]


@dataclass(frozen=True)
class ExportResult:
    """Result of an export operation."""

    success: bool
    url: str | None = None
    doc_id: str | None = None
    error: str | None = None


class GoogleDocsExporter:
    """Export meeting minutes to Google Docs via Google Drive HTML upload."""

    def __init__(self, cfg: ExportGoogleDocsConfig) -> None:
        self._cfg = cfg
        self._service: Any = None

    def _build_service(self) -> Any:
        """Build and cache the Google Drive API v3 service client."""
        if self._service is not None:
            return self._service
        creds_path = Path(self._cfg.credentials_path)
        if not creds_path.exists():
            raise ExportError(f"Credentials file not found: {creds_path}")

        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build

        creds = Credentials.from_service_account_file(
            str(creds_path), scopes=_SCOPES
        )
        self._service = build(
            "drive", "v3", credentials=creds, cache_discovery=False
        )
        return self._service

    def _md_to_html(self, minutes_md: str) -> str:
        """Convert Markdown to HTML."""
        import markdown

        md = markdown.Markdown(extensions=["tables", "fenced_code"])
        body = md.convert(minutes_md)
        return (
            "<!DOCTYPE html>"
            '<html><head><meta charset="utf-8"></head>'
            f"<body>{body}</body></html>"
        )

    def _upload_as_doc_sync(self, html: str, title: str) -> tuple[str, str]:
        """Upload HTML to Drive as a Google Docs document (synchronous)."""
        from googleapiclient.http import MediaInMemoryUpload

        service = self._build_service()
        media = MediaInMemoryUpload(
            html.encode("utf-8"),
            mimetype="text/html",
            resumable=False,
        )
        file_metadata = {
            "name": title,
            "mimeType": "application/vnd.google-apps.document",
            "parents": [self._cfg.folder_id],
        }
        result = (
            service.files()
            .create(
                body=file_metadata,
                media_body=media,
                fields="id, webViewLink",
            )
            .execute()
        )
        return result["id"], result["webViewLink"]

    async def export(
        self,
        minutes_md: str,
        title: str,
        metadata: dict[str, str] | None = None,
    ) -> ExportResult:
        """Export minutes to Google Docs. Never raises - returns ExportResult."""
        html = self._md_to_html(minutes_md)
        last_error = None

        for attempt in range(1, self._cfg.max_retries + 1):
            try:
                t0 = time.monotonic()
                doc_id, url = await asyncio.to_thread(
                    self._upload_as_doc_sync, html, title
                )
                elapsed = time.monotonic() - t0
                logger.info(
                    "Minutes exported to Google Docs in %.1fs: %s",
                    elapsed,
                    url,
                )
                return ExportResult(success=True, url=url, doc_id=doc_id)
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Export attempt %d/%d failed: %s",
                    attempt,
                    self._cfg.max_retries,
                    exc,
                )
                # Check for non-retryable HTTP errors
                if hasattr(exc, "resp") and hasattr(exc.resp, "status"):
                    status = exc.resp.status
                    if 400 <= status < 500 and status not in (403, 429):
                        break
                if attempt < self._cfg.max_retries:
                    delay = 2 ** (attempt - 1)
                    await asyncio.sleep(delay)

        logger.error(
            "Export failed after %d attempts: %s",
            self._cfg.max_retries,
            last_error,
        )
        return ExportResult(success=False, error=last_error)
