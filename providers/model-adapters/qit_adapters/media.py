import base64
import io
from fractions import Fraction


def media_bytes(case, prefix):
    value = case.get("media")
    if not isinstance(value, str) or not value.startswith(f"data:{prefix}"):
        raise ValueError(f"media must be an inline {prefix} data URL")
    metadata, separator, encoded = value.partition(",")
    if (
        not separator
        or not metadata.endswith(";base64")
        or len(encoded) > 48 * 1024 * 1024
    ):
        raise ValueError("media must be bounded base64 data")
    try:
        return base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise ValueError("media must be bounded base64 data") from error


def image_from_case(case):
    from PIL import Image

    data = media_bytes(case, "image/")
    image = Image.open(io.BytesIO(data))
    if image.width * image.height > 16_000_000:
        raise ValueError("input image exceeds sixteen million pixels")
    image.load()
    return image.convert("RGB")


def audio_from_case(case, sampling_rate):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    data = media_bytes(case, "audio/")
    with sf.SoundFile(io.BytesIO(data)) as audio:
        if audio.frames > audio.samplerate * 30:
            raise ValueError("CPU speech cases must be at most thirty seconds")
        samples = audio.read(dtype="float32", always_2d=True).mean(axis=1)
        original_rate = audio.samplerate
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError("audio must contain finite samples")
    if original_rate != sampling_rate:
        ratio = Fraction(sampling_rate, original_rate)
        samples = resample_poly(samples, ratio.numerator, ratio.denominator).astype(
            "float32"
        )
    return samples
