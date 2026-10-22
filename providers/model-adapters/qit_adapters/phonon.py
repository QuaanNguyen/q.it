import array
import base64
import math
import sys

from . import __version__

LANGUAGES = {"en", "fr", "de", "es", "pt", "none"}


def pcm16(values):
    samples = array.array(
        "h",
        (
            max(-32768, min(32767, int(sample * 32767.0)))
            for sample in values
        ),
    )
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


class PhononRuntime:
    def __init__(self, profile, directory, ptts_module=None):
        if ptts_module is None:
            import ptts as ptts_module
        self.profile = profile
        self.directory = directory
        self.ptts = ptts_module
        self.engines = {}
        self.engine = self.engine_for("en")
        self.sample_rate = int(self.engine.sample_rate)
        self.voices = sorted(str(voice) for voice in self.engine.voices)
        self.configuration = {
            "runtime": "ptts",
            "version": getattr(self.ptts, "__version__", None),
            "sample_rate_hz": self.sample_rate,
            "voices": self.voices,
            "normalization": "publisher ptts text normalization",
            "streaming": "publisher ptts float32 PCM chunks",
        }
        self.preparation = None
        self.retrieval = None
        self.extra_tensors = {}
        self.logical_parameters = []

    def engine_for(self, language):
        engine = self.engines.get(language)
        if engine is None:
            engine = self.ptts.TTS(str(self.directory), lang=language)
            self.engines[language] = engine
        return engine

    def validate_request(self, payload):
        case = payload["case"]
        speech = case.get("speech") or {}
        if not isinstance(speech, dict):
            raise ValueError("speech settings must be an object")
        if set(speech) - {"language", "voice", "max_audio_seconds"}:
            raise ValueError("speech settings contain unsupported fields")
        prompt = case.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("text to speech requires a nonempty prompt")
        if len(prompt) > 4000:
            raise ValueError("text to speech prompt must be at most 4000 characters")
        language = speech.get("language")
        if language is not None:
            if not isinstance(language, str) or language not in LANGUAGES:
                raise ValueError(
                    "speech language must be en, fr, de, es, pt, or none"
                )
        voice = speech.get("voice")
        if voice is not None and (
            not isinstance(voice, str) or not voice.strip()
        ):
            raise ValueError("speech voice must be a nonempty name")
        if voice is not None and voice not in self.voices:
            raise ValueError(
                "speech voice must already exist in the local checkpoint"
            )
        limit = speech.get("max_audio_seconds")
        if limit is not None:
            if (
                not isinstance(limit, (int, float))
                or isinstance(limit, bool)
                or not math.isfinite(limit)
                or not 1.0 <= limit <= 120.0
            ):
                raise ValueError("max_audio_seconds must be between 1 and 120")

    def invoke(self, case, max_tokens):
        self.validate_request({"case": case})
        speech = case.get("speech") or {}
        language = speech.get("language") or "en"
        voice = speech.get("voice")
        limit = speech.get("max_audio_seconds")
        engine = self.engine_for(language)
        sample_rate = int(engine.sample_rate)
        if sample_rate != self.sample_rate:
            raise ValueError("ptts sample rate changed between requests")
        chunks = 0
        samples = 0
        with engine.stream(case["prompt"], voice=voice) as audio:
            for chunk in audio:
                values = [float(sample) for sample in chunk]
                if not values:
                    continue
                if not all(math.isfinite(sample) for sample in values):
                    raise ValueError("ptts produced nonfinite audio samples")
                samples += len(values)
                if limit is not None and samples / sample_rate > limit:
                    raise ValueError(
                        f"synthesized audio exceeded the {limit:g}s case limit"
                    )
                data = pcm16(values)
                chunks += 1
                yield {
                    "type": "audio",
                    "pcm_base64": base64.b64encode(data).decode("ascii"),
                    "sample_rate_hz": sample_rate,
                }
        if not samples:
            raise ValueError("ptts produced no audio")
        yield {
            "type": "result",
            "audio_duration_seconds": samples / sample_rate,
            "sample_rate_hz": sample_rate,
            "audio_bytes": samples * 2,
            "speech_routing": {
                "route": "phonon",
                "runtime": "ptts",
                "adapter_version": __version__,
                "language": language,
                "voice": voice or "default",
                "chunks": chunks,
                "sample_rate_hz": sample_rate,
            },
        }
