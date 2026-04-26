#!/usr/bin/env python3
"""
Bulk-generate "Hey Harold" positive samples via Piper TTS.

Why Piper alongside ElevenLabs:
  - ElevenLabs gives us Italian-accent variety (Elena coverage, ~2000 samples,
    quota-limited).
  - Piper libritts-r gives us US-English variety: 904 distinct speakers, free,
    fast, runs entirely local. Bulk positive class for the model — ~3000-5000
    samples spanning gender/age/regional US accents.

  Together: ElevenLabs ~2000 (Italian-accent) + Piper ~3000 (US-English)
  + 62 real Alfie recordings + augmentation at training time.

What it produces:
  - WAV files in ./output/piper_samples/ at 16 kHz mono (matches Echo I2S).
  - Each file passes through the same trim+pad as elevenlabs_generate.py so
    head/tail silence is uniform.

Auto-downloads the voice model on first run (~75 MB). Subsequent runs reuse
the cached file at ./output/piper_voices/libritts-r-medium.onnx.

Usage:
  py -3 piper_generate.py --count 3000
  py -3 piper_generate.py --count 100 --speakers 0,42,123,500  # specific speakers
  py -3 piper_generate.py --list-speakers
"""
import argparse
import json
import random
import sys
import urllib.request
import wave
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "piper_samples"
VOICES_DIR = Path(__file__).resolve().parent.parent / "piper_voices"

VOICE_NAME = "libritts-r-medium"
HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium"
MODEL_URL = f"{HF_BASE}/en_US-libritts_r-medium.onnx"
CONFIG_URL = f"{HF_BASE}/en_US-libritts_r-medium.onnx.json"

# Plain phrases — Piper has no bracket-tag support like ElevenLabs v3, so
# variation comes from the 904-speaker pool + per-speaker prosody.
PHRASE_VARIANTS = [
    "Hey Harold",
    "Hey, Harold.",
    "Hey Harold!",
    "Hey, Harold?",
    "Hey Harold...",
    "hey harold",
    "Hey Harold,",
]


def ensure_voice_model() -> tuple[Path, Path]:
    """Download the Piper voice model + config if not cached."""
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    model_path = VOICES_DIR / f"{VOICE_NAME}.onnx"
    config_path = VOICES_DIR / f"{VOICE_NAME}.onnx.json"

    for url, dest in [(MODEL_URL, model_path), (CONFIG_URL, config_path)]:
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"Downloading {dest.name} ({url})...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  {dest.stat().st_size // 1024 // 1024} MB")
        except Exception as e:
            sys.exit(f"  FAILED: {e}")

    return model_path, config_path


def load_speaker_list(config_path: Path) -> list[int]:
    """Read available speaker IDs from the model config."""
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    speaker_id_map = cfg.get("speaker_id_map", {})
    if not speaker_id_map:
        # Single-speaker model
        return [0]
    return sorted(int(v) for v in speaker_id_map.values())


def _trim_and_pad(pcm_bytes: bytes, sample_rate: int, head_ms: int,
                  tail_ms: int, silence_threshold: int = 400) -> bytes:
    """Mirror of elevenlabs_generate._trim_and_pad — keeps both pipelines
    producing samples with identical 80ms head/tail format."""
    import array
    samples = array.array("h")
    samples.frombytes(pcm_bytes)
    if not samples:
        return pcm_bytes
    win = max(1, sample_rate // 100)
    start_i = 0
    for i in range(0, len(samples) - win, win):
        if max(abs(s) for s in samples[i:i + win]) >= silence_threshold:
            start_i = max(0, i - win)
            break
    end_i = len(samples)
    for i in range(len(samples) - win, 0, -win):
        if max(abs(s) for s in samples[i:i + win]) >= silence_threshold:
            end_i = min(len(samples), i + 2 * win)
            break
    if end_i <= start_i:
        return pcm_bytes
    trimmed = samples[start_i:end_i]
    head = array.array("h", [0] * ((head_ms * sample_rate) // 1000))
    tail = array.array("h", [0] * ((tail_ms * sample_rate) // 1000))
    return (head + trimmed + tail).tobytes()


def _resample_22050_to_16000(pcm_bytes: bytes) -> bytes:
    """Linear-interp downsample 22050 → 16000 Hz, 16-bit mono."""
    import array
    src = array.array("h")
    src.frombytes(pcm_bytes)
    if not src:
        return pcm_bytes
    ratio = 22050 / 16000
    out_len = int(len(src) / ratio)
    dst = array.array("h", [0] * out_len)
    for i in range(out_len):
        src_pos = i * ratio
        i0 = int(src_pos)
        if i0 + 1 >= len(src):
            dst[i] = src[-1]
        else:
            frac = src_pos - i0
            dst[i] = int(src[i0] * (1 - frac) + src[i0 + 1] * frac)
    return dst.tobytes()


def _write_pcm_wav(path: Path, pcm_bytes: bytes, sample_rate: int = 16000) -> None:
    import struct
    n_samples = len(pcm_bytes) // 2
    byte_rate = sample_rate * 2
    riff_size = 36 + len(pcm_bytes)
    header = b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
    fmt = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, 2, 16)
    data = b"data" + struct.pack("<I", len(pcm_bytes)) + pcm_bytes
    path.write_bytes(header + fmt + data)


def synthesize_one(piper_voice, text: str, speaker_id: int) -> bytes:
    """Return raw 16-bit mono PCM at 22050 Hz for the given text+speaker."""
    from piper.config import SynthesisConfig
    cfg = SynthesisConfig(speaker_id=speaker_id, normalize_audio=True)
    chunks = list(piper_voice.synthesize(text, syn_config=cfg))
    return b"".join(c.audio_int16_bytes for c in chunks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=3000,
                    help="total samples to generate (spread across speaker pool)")
    ap.add_argument("--speakers", type=str,
                    help="comma-separated speaker IDs to use (default: all)")
    ap.add_argument("--list-speakers", action="store_true",
                    help="show available speaker IDs and exit")
    ap.add_argument("--label", default="piper",
                    help="output filename prefix")
    args = ap.parse_args()

    model_path, config_path = ensure_voice_model()
    all_speakers = load_speaker_list(config_path)
    print(f"Voice model has {len(all_speakers)} speakers")

    if args.list_speakers:
        print("Speaker IDs:", all_speakers[:20], "..." if len(all_speakers) > 20 else "")
        return

    if args.speakers:
        speakers = [int(s) for s in args.speakers.split(",")]
    else:
        speakers = all_speakers

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading Piper voice...")
    from piper import PiperVoice
    voice = PiperVoice.load(model_path)

    samples_per_speaker = max(1, args.count // len(speakers))
    print(f"Generating {samples_per_speaker} samples × {len(speakers)} speakers")
    print(f"Output: {OUTPUT_DIR}\n")

    saved = 0
    rng = random.Random(42)
    import time
    started = time.time()
    for spk in speakers:
        for n in range(samples_per_speaker):
            phrase = PHRASE_VARIANTS[n % len(PHRASE_VARIANTS)]
            try:
                pcm_22k = synthesize_one(voice, phrase, spk)
                pcm_16k = _resample_22050_to_16000(pcm_22k)
                pcm_16k = _trim_and_pad(pcm_16k, 16000, head_ms=80, tail_ms=80)
                wav_path = OUTPUT_DIR / f"{args.label}_spk{spk:04d}_{n:02d}.wav"
                _write_pcm_wav(wav_path, pcm_16k, 16000)
                saved += 1
                if saved % 100 == 0:
                    elapsed = time.time() - started
                    rate = saved / elapsed
                    eta = (samples_per_speaker * len(speakers) - saved) / rate
                    print(f"  {saved}/{samples_per_speaker * len(speakers)} ({rate:.1f}/s, ETA {eta:.0f}s)")
            except Exception as e:
                print(f"  ERROR spk={spk} n={n}: {e}")

    print(f"\nDone. {saved} written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
