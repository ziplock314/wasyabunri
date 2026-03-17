"""Tests for src/transcriber_api.py -- OpenAI Speech-to-Text backend."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.audio_source import SpeakerAudio, SpeakerInfo
from src.errors import TranscriptionError
from src.transcriber import Segment
from src.transcriber_api import TranscriberAPI


# ---------------------------------------------------------------------------
# Helpers -- mock WhisperConfig with the new API fields
# ---------------------------------------------------------------------------

def _make_cfg(
    *,
    language: str = "ja",
    api_model: str = "whisper-1",
    api_max_retries: int = 2,
    api_timeout_sec: int = 300,
    backend: str = "api",
) -> SimpleNamespace:
    """Build a config-like object with the fields TranscriberAPI needs.

    Uses SimpleNamespace so that the tests work regardless of whether the
    new fields have been added to the real WhisperConfig dataclass yet.
    """
    return SimpleNamespace(
        model="large-v3",
        language=language,
        device="cuda",
        compute_type="float16",
        beam_size=5,
        vad_filter=True,
        backend=backend,
        api_model=api_model,
        api_max_retries=api_max_retries,
        api_timeout_sec=api_timeout_sec,
    )


def _mock_segment(start: float, end: float, text: str) -> SimpleNamespace:
    """Create a mock response segment matching the OpenAI verbose_json shape."""
    return SimpleNamespace(start=start, end=end, text=text)


def _mock_response(segments: list | None = None) -> SimpleNamespace:
    """Create a mock OpenAI transcription response."""
    return SimpleNamespace(segments=segments)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestTranscriberAPILifecycle:
    def test_not_loaded_by_default(self) -> None:
        t = TranscriberAPI(_make_cfg())
        assert t.is_loaded is False

    def test_load_model_creates_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        t = TranscriberAPI(_make_cfg())

        with patch("src.transcriber_api.openai.OpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            t.load_model()

            assert t.is_loaded is True
            mock_cls.assert_called_once_with(api_key="sk-test-key", timeout=300)

    def test_load_model_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        t = TranscriberAPI(_make_cfg())

        with patch("src.transcriber_api.openai.OpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            t.load_model()
            t.load_model()  # second call should be a no-op
            mock_cls.assert_called_once()

    def test_load_model_no_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        t = TranscriberAPI(_make_cfg())

        with pytest.raises(TranscriptionError, match="OPENAI_API_KEY"):
            t.load_model()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestTranscriberAPIProperties:
    def test_backend_name(self) -> None:
        t = TranscriberAPI(_make_cfg())
        assert t.backend_name == "api"

    def test_model_name(self) -> None:
        t = TranscriberAPI(_make_cfg(api_model="whisper-1"))
        assert t.model_name == "whisper-1"

    def test_model_name_custom(self) -> None:
        t = TranscriberAPI(_make_cfg(api_model="whisper-2-preview"))
        assert t.model_name == "whisper-2-preview"


# ---------------------------------------------------------------------------
# transcribe_file
# ---------------------------------------------------------------------------


class TestTranscribeFile:
    def test_transcribe_file_before_load_raises(self, tmp_path: Path) -> None:
        t = TranscriberAPI(_make_cfg())
        dummy = tmp_path / "audio.aac"
        dummy.write_bytes(b"\x00" * 100)
        with pytest.raises(TranscriptionError, match="not loaded"):
            t.transcribe_file(dummy, "speaker")

    def test_transcribe_file_missing_file(self) -> None:
        t = TranscriberAPI(_make_cfg())
        t._client = MagicMock()
        with pytest.raises(TranscriptionError, match="not found"):
            t.transcribe_file(Path("/nonexistent/audio.aac"), "speaker")

    def test_file_too_large_raises(self, tmp_path: Path) -> None:
        t = TranscriberAPI(_make_cfg())
        t._client = MagicMock()

        big_file = tmp_path / "huge.aac"
        # Create a file just over 25 MB
        big_file.write_bytes(b"\x00" * (25 * 1024 * 1024 + 1))

        with pytest.raises(TranscriptionError, match="25 MB"):
            t.transcribe_file(big_file, "speaker")

    def test_transcribe_file_success(self, tmp_path: Path) -> None:
        t = TranscriberAPI(_make_cfg())

        mock_client = MagicMock()
        response = _mock_response(
            segments=[
                _mock_segment(0.0, 2.5, "  Hello world  "),
                _mock_segment(3.0, 5.0, "How are you?"),
            ]
        )
        mock_client.audio.transcriptions.create.return_value = response
        t._client = mock_client

        audio = tmp_path / "test.aac"
        audio.write_bytes(b"\x00" * 100)

        result = t.transcribe_file(audio, "Alice")

        assert len(result) == 2
        assert all(isinstance(s, Segment) for s in result)

        assert result[0].start == 0.0
        assert result[0].end == 2.5
        assert result[0].text == "Hello world"  # stripped
        assert result[0].speaker == "Alice"

        assert result[1].start == 3.0
        assert result[1].end == 5.0
        assert result[1].text == "How are you?"
        assert result[1].speaker == "Alice"

    def test_transcribe_file_empty_segments_filtered(self, tmp_path: Path) -> None:
        t = TranscriberAPI(_make_cfg())

        mock_client = MagicMock()
        response = _mock_response(
            segments=[
                _mock_segment(0.0, 1.0, "   "),       # blank after strip
                _mock_segment(1.0, 2.0, ""),           # empty
                _mock_segment(2.0, 3.0, "  Valid  "),  # kept
                _mock_segment(3.0, 4.0, "\t\n"),       # whitespace only
            ]
        )
        mock_client.audio.transcriptions.create.return_value = response
        t._client = mock_client

        audio = tmp_path / "test.aac"
        audio.write_bytes(b"\x00" * 100)

        result = t.transcribe_file(audio, "Bob")
        assert len(result) == 1
        assert result[0].text == "Valid"

    def test_transcribe_file_no_segments_attribute(self, tmp_path: Path) -> None:
        """Gracefully handle a response with no segments attribute."""
        t = TranscriberAPI(_make_cfg())

        mock_client = MagicMock()
        # Response object without segments attribute
        mock_client.audio.transcriptions.create.return_value = SimpleNamespace()
        t._client = mock_client

        audio = tmp_path / "test.aac"
        audio.write_bytes(b"\x00" * 100)

        result = t.transcribe_file(audio, "Bob")
        assert result == []


# ---------------------------------------------------------------------------
# Language handling
# ---------------------------------------------------------------------------


class TestLanguageHandling:
    def test_auto_language_passes_none(self, tmp_path: Path) -> None:
        t = TranscriberAPI(_make_cfg(language="auto"))

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = _mock_response(
            segments=[]
        )
        t._client = mock_client

        audio = tmp_path / "test.aac"
        audio.write_bytes(b"\x00" * 100)

        t.transcribe_file(audio, "speaker")

        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert kwargs["language"] is None

    def test_explicit_language_passes_through(self, tmp_path: Path) -> None:
        t = TranscriberAPI(_make_cfg(language="ja"))

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = _mock_response(
            segments=[]
        )
        t._client = mock_client

        audio = tmp_path / "test.aac"
        audio.write_bytes(b"\x00" * 100)

        t.transcribe_file(audio, "speaker")

        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert kwargs["language"] == "ja"

    def test_english_language_passes_through(self, tmp_path: Path) -> None:
        t = TranscriberAPI(_make_cfg(language="en"))

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = _mock_response(
            segments=[]
        )
        t._client = mock_client

        audio = tmp_path / "test.aac"
        audio.write_bytes(b"\x00" * 100)

        t.transcribe_file(audio, "speaker")

        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert kwargs["language"] == "en"


# ---------------------------------------------------------------------------
# transcribe_all
# ---------------------------------------------------------------------------


class TestTranscribeAll:
    def test_transcribe_all_combines_tracks(self, tmp_path: Path) -> None:
        t = TranscriberAPI(_make_cfg())

        # Create two audio files
        f1 = tmp_path / "1-alice.aac"
        f1.write_bytes(b"\x00" * 100)
        f2 = tmp_path / "2-bob.aac"
        f2.write_bytes(b"\x00" * 100)

        response_alice = _mock_response(
            segments=[
                _mock_segment(0.0, 1.0, "Hello from Alice"),
                _mock_segment(1.5, 3.0, "Still Alice"),
            ]
        )
        response_bob = _mock_response(
            segments=[
                _mock_segment(0.5, 2.0, "Bob here"),
            ]
        )

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = [
            response_alice,
            response_bob,
        ]
        t._client = mock_client

        tracks = [
            SpeakerAudio(
                speaker=SpeakerInfo(track=1, username="alice", user_id=1),
                file_path=f1,
            ),
            SpeakerAudio(
                speaker=SpeakerInfo(track=2, username="bob", user_id=2),
                file_path=f2,
            ),
        ]

        result = t.transcribe_all(tracks)

        assert len(result) == 3
        assert mock_client.audio.transcriptions.create.call_count == 2

        # First two segments are Alice's
        assert result[0].speaker == "alice"
        assert result[0].text == "Hello from Alice"
        assert result[1].speaker == "alice"
        assert result[1].text == "Still Alice"

        # Third is Bob's
        assert result[2].speaker == "bob"
        assert result[2].text == "Bob here"

    def test_transcribe_all_empty_tracks(self) -> None:
        t = TranscriberAPI(_make_cfg())
        t._client = MagicMock()
        result = t.transcribe_all([])
        assert result == []


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


class TestRetryBehaviour:
    def test_api_error_retries(self, tmp_path: Path) -> None:
        """Transient RateLimitError should be retried up to api_max_retries times."""
        t = TranscriberAPI(_make_cfg(api_max_retries=2))

        audio = tmp_path / "test.aac"
        audio.write_bytes(b"\x00" * 100)

        rate_limit_exc = _make_rate_limit_error()
        success_response = _mock_response(
            segments=[_mock_segment(0.0, 1.0, "Recovered")]
        )

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = [
            rate_limit_exc,
            success_response,
        ]
        t._client = mock_client

        with patch("src.transcriber_api.time.sleep"):
            result = t.transcribe_file(audio, "speaker")

        assert len(result) == 1
        assert result[0].text == "Recovered"
        assert mock_client.audio.transcriptions.create.call_count == 2

    def test_api_connection_error_retries(self, tmp_path: Path) -> None:
        """APIConnectionError should also trigger retries."""
        t = TranscriberAPI(_make_cfg(api_max_retries=1))

        audio = tmp_path / "test.aac"
        audio.write_bytes(b"\x00" * 100)

        import openai as _openai

        conn_exc = _openai.APIConnectionError(request=MagicMock())
        success_response = _mock_response(
            segments=[_mock_segment(0.0, 1.0, "OK")]
        )

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = [
            conn_exc,
            success_response,
        ]
        t._client = mock_client

        with patch("src.transcriber_api.time.sleep"):
            result = t.transcribe_file(audio, "speaker")

        assert len(result) == 1
        assert result[0].text == "OK"

    def test_server_error_retries(self, tmp_path: Path) -> None:
        """5xx APIStatusError should trigger retries."""
        t = TranscriberAPI(_make_cfg(api_max_retries=1))

        audio = tmp_path / "test.aac"
        audio.write_bytes(b"\x00" * 100)

        server_exc = _make_api_status_error(status_code=500)
        success_response = _mock_response(
            segments=[_mock_segment(0.0, 1.0, "OK")]
        )

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = [
            server_exc,
            success_response,
        ]
        t._client = mock_client

        with patch("src.transcriber_api.time.sleep"):
            result = t.transcribe_file(audio, "speaker")

        assert len(result) == 1
        assert result[0].text == "OK"

    def test_client_error_does_not_retry(self, tmp_path: Path) -> None:
        """4xx (non-429) APIStatusError should NOT be retried."""
        t = TranscriberAPI(_make_cfg(api_max_retries=2))

        audio = tmp_path / "test.aac"
        audio.write_bytes(b"\x00" * 100)

        client_exc = _make_api_status_error(status_code=400)

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = client_exc
        t._client = mock_client

        with pytest.raises(TranscriptionError, match="OpenAI API error"):
            t.transcribe_file(audio, "speaker")

        # Should have been called exactly once -- no retries
        assert mock_client.audio.transcriptions.create.call_count == 1

    def test_all_retries_exhausted_raises(self, tmp_path: Path) -> None:
        """When all retries are exhausted, raise TranscriptionError."""
        t = TranscriberAPI(_make_cfg(api_max_retries=1))

        audio = tmp_path / "test.aac"
        audio.write_bytes(b"\x00" * 100)

        rate_limit_exc = _make_rate_limit_error()

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = rate_limit_exc
        t._client = mock_client

        with patch("src.transcriber_api.time.sleep"):
            with pytest.raises(TranscriptionError, match="failed after"):
                t.transcribe_file(audio, "speaker")

        # 1 initial + 1 retry = 2 total
        assert mock_client.audio.transcriptions.create.call_count == 2

    def test_retry_uses_exponential_backoff(self, tmp_path: Path) -> None:
        """Verify sleep is called with exponential backoff durations."""
        t = TranscriberAPI(_make_cfg(api_max_retries=3))

        audio = tmp_path / "test.aac"
        audio.write_bytes(b"\x00" * 100)

        rate_limit_exc = _make_rate_limit_error()

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = rate_limit_exc
        t._client = mock_client

        with patch("src.transcriber_api.time.sleep") as mock_sleep:
            with pytest.raises(TranscriptionError):
                t.transcribe_file(audio, "speaker")

        # 4 attempts, 3 sleeps between them (not after the last)
        assert mock_sleep.call_count == 3
        sleep_args = [call.args[0] for call in mock_sleep.call_args_list]
        assert sleep_args == [1, 2, 4]  # 2^0, 2^1, 2^2


# ---------------------------------------------------------------------------
# API call parameters
# ---------------------------------------------------------------------------


class TestAPICallParameters:
    def test_api_call_parameters(self, tmp_path: Path) -> None:
        """Verify all parameters passed to the OpenAI API."""
        t = TranscriberAPI(_make_cfg(api_model="whisper-1", language="ja"))

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = _mock_response(
            segments=[]
        )
        t._client = mock_client

        audio = tmp_path / "test.aac"
        audio.write_bytes(b"\x00" * 100)

        t.transcribe_file(audio, "speaker")

        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert kwargs["model"] == "whisper-1"
        assert kwargs["language"] == "ja"
        assert kwargs["response_format"] == "verbose_json"
        assert kwargs["timestamp_granularities"] == ["segment"]


# ---------------------------------------------------------------------------
# Helpers for constructing openai exception instances
# ---------------------------------------------------------------------------


def _make_rate_limit_error() -> Exception:
    """Create an openai.RateLimitError with minimal plumbing."""
    import openai as _openai

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    return _openai.RateLimitError(
        message="Rate limit exceeded",
        response=mock_response,
        body=None,
    )


def _make_api_status_error(status_code: int) -> Exception:
    """Create an openai.APIStatusError with the given status code."""
    import openai as _openai

    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.headers = {}
    return _openai.APIStatusError(
        message=f"Error {status_code}",
        response=mock_response,
        body=None,
    )
