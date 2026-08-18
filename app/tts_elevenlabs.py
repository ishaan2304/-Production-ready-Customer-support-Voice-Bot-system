"""
Enhanced TTS Module with ElevenLabs support.
ElevenLabs provides natural, human-like voices.
Falls back to gTTS if ElevenLabs is unavailable.
"""
import io
import os
import time
from typing import Optional, Dict, Any

from app.config import get_config
from app.exceptions import TTSError
from app.logger import get_logger

logger = get_logger(__name__)

# ElevenLabs voice presets
VOICE_PRESETS = {
    "rachel":  "21m00Tcm4TlvDq8ikWAM",   # Calm, professional female
    "adam":    "pNInz6obpgDQGcFmaJgB",    # Professional male
    "bella":   "EXAVITQu4vr4xnSDxMaL",   # Friendly female
    "josh":    "TxGEqnHWrfWFTfGW9XjX",   # Conversational male
}


class ElevenLabsTTS:
    """
    ElevenLabs text-to-speech with gTTS fallback.
    Provides natural, human-like voice synthesis.
    """

    def __init__(self):
        self.config = get_config().tts
        self._client = None
        self._loaded = False
        self._use_elevenlabs = False
        self._voice_id = None

    def _lazy_load(self) -> None:
        if self._loaded:
            return
        try:
            from elevenlabs.client import ElevenLabs
            from dotenv import load_dotenv
            load_dotenv()

            api_key = os.getenv("ELEVENLABS_API_KEY")
            if not api_key or api_key == "your_elevenlabs_api_key_here":
                logger.warning("ElevenLabs API key not set — using gTTS fallback")
                self._use_elevenlabs = False
                self._loaded = True
                return

            self._client = ElevenLabs(api_key=api_key)
            self._voice_id = os.getenv(
                "ELEVENLABS_VOICE_ID",
                VOICE_PRESETS["rachel"]
            )
            self._use_elevenlabs = True
            self._loaded = True
            logger.info(f"ElevenLabs TTS ready (voice: {self._voice_id})")

        except ImportError:
            logger.warning("elevenlabs package not installed — using gTTS fallback")
            self._use_elevenlabs = False
            self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def engine_name(self) -> str:
        return "elevenlabs" if self._use_elevenlabs else "gtts"

    def synthesize(
        self,
        text: str,
        language: str = "en",
        slow: bool = False,
        voice_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize text to speech.

        Args:
            text: Text to convert to speech
            language: Language code
            slow: Use slower speech rate
            voice_id: Override ElevenLabs voice ID

        Returns:
            Dict with audio_bytes, format, engine_used
        """
        if not text or not text.strip():
            raise TTSError("Empty text provided for synthesis")

        self._lazy_load()
        start = time.perf_counter()
        clean_text = self._clean_text(text)

        try:
            if self._use_elevenlabs:
                audio_bytes = self._synthesize_elevenlabs(
                    clean_text, voice_id or self._voice_id
                )
                audio_format = "mp3"
            else:
                audio_bytes = self._synthesize_gtts(clean_text, language, slow)
                audio_format = "mp3"

            elapsed = (time.perf_counter() - start) * 1000
            word_count = len(clean_text.split())
            duration_estimate = (word_count / 150) * 60

            logger.debug(
                f"TTS ({self.engine_name}) synthesized {len(clean_text)} chars "
                f"in {elapsed:.1f}ms"
            )

            return {
                "audio_bytes": audio_bytes,
                "format": audio_format,
                "engine": self.engine_name,
                "language": language,
                "duration_estimate_seconds": round(duration_estimate, 1),
                "processing_time_ms": round(elapsed, 2),
            }

        except TTSError:
            raise
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}", exc_info=True)
            raise TTSError("Speech synthesis failed", str(e))

    def _synthesize_elevenlabs(self, text: str, voice_id: str) -> bytes:
        """Synthesize using ElevenLabs API."""
        try:
            audio_generator = self._client.generate(
                text=text,
                voice=voice_id,
                model="eleven_turbo_v2",  # Fast, high quality
            )
            # Convert generator to bytes
            audio_bytes = b"".join(audio_generator)
            return audio_bytes
        except Exception as e:
            logger.warning(f"ElevenLabs failed: {e}, falling back to gTTS")
            return self._synthesize_gtts(text, "en", False)

    def _synthesize_gtts(self, text: str, language: str, slow: bool) -> bytes:
        """Synthesize using gTTS as fallback."""
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang=language, slow=slow)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            return buf.read()
        except Exception as e:
            raise TTSError("gTTS synthesis failed", str(e))

    def _clean_text(self, text: str) -> str:
        """Clean text for TTS."""
        text = " ".join(text.split())
        text = text.replace("**", "").replace("*", "").replace("_", "")
        text = text.replace("[", "").replace("]", "")
        if len(text) > 2000:
            text = text[:2000] + "."
        return text.strip()

    def list_voices(self) -> list:
        """List available ElevenLabs voices."""
        if not self._use_elevenlabs:
            return [{"name": "gTTS Default", "voice_id": "gtts"}]
        try:
            response = self._client.voices.get_all()
            return [
                {"name": v.name, "voice_id": v.voice_id}
                for v in response.voices
            ]
        except Exception as e:
            logger.error(f"Failed to list voices: {e}")
            return list(VOICE_PRESETS.items())
