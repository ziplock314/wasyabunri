"""OpenAI Speech-to-Text API transcription backend.

Drop-in alternative to the local faster-whisper ``Transcriber``.
Uses the same public interface (duck typing) so it can be swapped
transparently via ``WhisperConfig.backend``.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import openai

from src.audio_source import SpeakerAudio
from src.config import WhisperConfig
from src.errors import TranscriptionError
from src.transcriber import Segment

logger = logging.getLogger(__name__)

# OpenAI Whisper API hard limit
_MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB

# Cost per minute for the whisper-1 model (USD)
_COST_PER_MINUTE = 0.006


class TranscriberAPI:
    """OpenAI Speech-to-Text API transcription backend.

    Implements the same public interface as :class:`src.transcriber.Transcriber`
    so that callers can use either backend interchangeably.
    """

    def __init__(self, cfg: WhisperConfig) -> None:
        self._cfg = cfg
        self._client: openai.OpenAI | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Create the OpenAI client.

        Reads ``OPENAI_API_KEY`` from the environment.
        Raises :class:`TranscriptionError` if the key is missing.
        Subsequent calls are no-ops.
        """
        if self._client is not None:
            return

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise TranscriptionError(
                "OPENAI_API_KEY environment variable is not set"
            )

        self._client = openai.OpenAI(
            api_key=api_key,
            timeout=self._cfg.api_timeout_sec,
        )
        logger.info(
            "OpenAI transcription client ready (model=%s)",
            self._cfg.api_model,
        )

    @property
    def is_loaded(self) -> bool:
        """Return ``True`` when the OpenAI client has been initialised."""
        return self._client is not None

    @property
    def backend_name(self) -> str:
        """Return ``'api'`` to distinguish from the local backend."""
        return "api"

    @property
    def model_name(self) -> str:
        """Return the configured API model name (e.g. ``'whisper-1'``)."""
        return self._cfg.api_model

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe_file(
        self, audio_path: Path, speaker_name: str
    ) -> list[Segment]:
        """Transcribe a single audio file via the OpenAI API.

        Parameters
        ----------
        audio_path:
            Path to the audio file (AAC, FLAC, MP3, WAV, etc.).
        speaker_name:
            Human-readable speaker label attached to every returned segment.

        Returns
        -------
        list[Segment]
            Timestamped segments with speaker attribution.

        Raises
        ------
        TranscriptionError
            If the client is not loaded, the file is missing / too large,
            or the API call fails after retries.
        """
        if self._client is None:
            raise TranscriptionError(
                "Client not loaded -- call load_model() first"
            )

        path = Path(audio_path)
        if not path.exists():
            raise TranscriptionError(f"Audio file not found: {path}")

        file_size = path.stat().st_size
        if file_size > _MAX_FILE_SIZE_BYTES:
            raise TranscriptionError(
                f"File {path.name} is {file_size / 1024 / 1024:.1f} MB, "
                f"exceeding the 25 MB API limit"
            )

        language = (
            None if self._cfg.language == "auto" else self._cfg.language
        )

        logger.info(
            "Transcribing %s via OpenAI API (speaker=%s, model=%s)",
            path.name,
            speaker_name,
            self._cfg.api_model,
        )
        t0 = time.monotonic()

        response = self._call_api_with_retry(path, language)

        segments: list[Segment] = []
        for s in getattr(response, "segments", None) or []:
            text = s.text.strip()
            if text:
                segments.append(
                    Segment(
                        start=s.start,
                        end=s.end,
                        text=text,
                        speaker=speaker_name,
                    )
                )

        elapsed = time.monotonic() - t0
        duration_min = (
            segments[-1].end / 60.0 if segments else 0.0
        )
        cost_estimate = duration_min * _COST_PER_MINUTE

        logger.info(
            "Transcribed %s: %d segments in %.1fs "
            "(duration=%.1fmin, est_cost=$%.4f)",
            path.name,
            len(segments),
            elapsed,
            duration_min,
            cost_estimate,
        )
        return segments

    def transcribe_all(
        self, tracks: list[SpeakerAudio]
    ) -> list[Segment]:
        """Transcribe all speaker audio tracks sequentially.

        Returns the combined (unsorted) list of segments from all tracks.
        """
        all_segments: list[Segment] = []

        for i, track in enumerate(tracks, 1):
            logger.info(
                "Transcribing speaker %d/%d: %s",
                i,
                len(tracks),
                track.speaker.username,
            )
            segments = self.transcribe_file(
                track.file_path,
                speaker_name=track.speaker.username,
            )
            all_segments.extend(segments)

        logger.info(
            "Transcription complete: %d total segments from %d speakers",
            len(all_segments),
            len(tracks),
        )
        return all_segments

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_api_with_retry(self, path: Path, language: str | None) -> object:
        """Call the OpenAI transcription endpoint with exponential backoff.

        Retries on transient errors (rate-limit, connection, 5xx) up to
        ``cfg.api_max_retries`` times.
        """
        max_retries = self._cfg.api_max_retries
        last_exc: Exception | None = None

        for attempt in range(1 + max_retries):
            try:
                with open(path, "rb") as f:
                    response = self._client.audio.transcriptions.create(
                        model=self._cfg.api_model,
                        file=f,
                        language=language,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"],
                    )
                return response

            except openai.RateLimitError as exc:
                last_exc = exc
                logger.warning(
                    "Rate limited (attempt %d/%d): %s",
                    attempt + 1,
                    1 + max_retries,
                    exc,
                )
            except openai.APIConnectionError as exc:
                last_exc = exc
                logger.warning(
                    "Connection error (attempt %d/%d): %s",
                    attempt + 1,
                    1 + max_retries,
                    exc,
                )
            except openai.APIStatusError as exc:
                if exc.status_code >= 500:
                    last_exc = exc
                    logger.warning(
                        "Server error %d (attempt %d/%d): %s",
                        exc.status_code,
                        attempt + 1,
                        1 + max_retries,
                        exc,
                    )
                else:
                    # 4xx (except 429 which is RateLimitError) -- not transient
                    raise TranscriptionError(
                        f"OpenAI API error for {path.name}: {exc}"
                    ) from exc
            except Exception as exc:
                raise TranscriptionError(
                    f"OpenAI API call failed for {path.name}: {exc}"
                ) from exc

            if attempt < max_retries:
                backoff = 2 ** attempt
                logger.info("Retrying in %ds ...", backoff)
                time.sleep(backoff)

        raise TranscriptionError(
            f"OpenAI API failed after {1 + max_retries} attempts "
            f"for {path.name}: {last_exc}"
        )
