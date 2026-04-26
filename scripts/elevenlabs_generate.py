#!/usr/bin/env python3
"""
Generate accent-diverse positive wake-word samples using ElevenLabs TTS.

Why this exists alongside the Piper sample generator (which Colab uses by default):
  - Piper's libritts-r is US-English only (~900 voices). Plenty of English variety
    but no native non-English-accent speakers.
  - ElevenLabs has 5000+ voices including native speakers of many languages
    speaking English with their natural accent. Three sub-modes:
      * `italian`  — Italian-accent voices from the public library
      * `cloned`   — your own voice clone (create in ElevenLabs UI first)
      * `diverse`  — hand-picked diverse voices across accents
  - Bracketed prosody tags ([whispered], [shouted], [casually], etc) get
    natural vocal-effort variation in the samples — only `eleven_v3` honours
    these reliably; v2 reads them as text.

Cost (ElevenLabs Creator tier, ~£11/mo):
  - 100k chars/mo included
  - A short wake phrase (10-15 chars) × 22 prosody templates × 2000 samples
    ≈ 60-80k chars — within the monthly allowance.
  - Don't run this on the Free tier — only 10k chars/mo, you'll burn it instantly.

Usage:
  ELEVENLABS_API_KEY=sk_xxx py -3 elevenlabs_generate.py \\
      --phrase "Hey Jeeves" --mode italian --count 2000
  py -3 elevenlabs_generate.py --phrase "Hey Jeeves" --list-italian-voices

Output: WAV files in ./elevenlabs_samples/<mode>/.
Drop these into your training bundle's generated_samples/ alongside Piper output.
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# Looks for an ELEVENLABS_API_KEY in (in order):
#   1. environment variable ELEVENLABS_API_KEY
#   2. ./.env at the repo root (next to this script's parent dir)
#   3. ~/.elevenlabs.env
# Create one of these — DO NOT commit your key. The repo .gitignore excludes .env.
import os

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "elevenlabs_samples"

# Prosody-tag templates — `{phrase}` is replaced with your wake word at runtime.
# Bracketed style cues like [whispered] are ElevenLabs v3 style directives that
# the model interprets as vocal-effort/manner. They do NOT add real acoustic
# variation (reverb, noise) — that's the training pipeline's job. They DO add
# behavioural variation across the speaker pool, which is what wake-word
# training needs.
PHRASE_VARIANT_TEMPLATES = [
    # Neutral baseline
    "{phrase}", "{phrase}.", "{phrase}!", "{phrase}...",
    # Casual / half-attention (most common real wake-word context)
    "[casually] {phrase}",
    "[casually, half-distracted] {phrase}",
    "[murmured] {phrase}.",
    "[under breath] {phrase}",
    # Across-the-room (raised vocal effort, no actual distance)
    "[calling from across the room] {phrase}!",
    "[calling out] {phrase}!",
    "[raised voice] {phrase}",
    # Quiet contexts (late night, sleeping partner nearby)
    "[whispered] {phrase}",
    "[quietly] {phrase}",
    "[hushed] {phrase}",
    # Sharp / commanding (urgent request)
    "[sharply] {phrase}",
    "[urgently] {phrase}!",
    "[clipped] {phrase}",
    # Sleepy / relaxed (morning, evening)
    "[sleepy] {phrase}",
    "[yawning slightly] {phrase}",
    "[relaxed] {phrase}.",
    # Mumble / under-articulated (real-world degradation)
    "[mumbled] {phrase}",
    "[slurred slightly] {phrase}",
]


def load_api_key() -> str:
    # 1. environment
    if key := os.environ.get("ELEVENLABS_API_KEY"):
        return key.strip()
    # 2. .env files in priority order
    for env_path in (REPO_ROOT / ".env",
                     Path.home() / ".elevenlabs.env"):
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("ELEVENLABS_API_KEY="):
                    return line.split("=", 1)[1].strip()
    sys.exit(
        "ERROR: ELEVENLABS_API_KEY not set.\n"
        "  Set it via: export ELEVENLABS_API_KEY=sk_xxx\n"
        f"  Or in: {REPO_ROOT / '.env'}\n"
        "  Or in: ~/.elevenlabs.env"
    )


def build_phrase_variants(phrase: str) -> list[str]:
    """Render every template with the given wake phrase substituted in."""
    return [t.format(phrase=phrase) for t in PHRASE_VARIANT_TEMPLATES]


def list_italian_voices(api_key: str) -> list[dict]:
    """Query the ElevenLabs public Voice Library for Italian-accent voices."""
    try:
        from elevenlabs import ElevenLabs
    except ImportError:
        sys.exit("Install: py -3 -m pip install elevenlabs")

    client = ElevenLabs(api_key=api_key)

    print("Querying ElevenLabs Voice Library for Italian-accent voices...")
    # The shared library exposes accent/language filters directly. Italian
    # speakers reading English are tagged language="en" + accent="italian";
    # native Italian voices are language="it". We want both — they all give
    # us natural Italian-accented English when prompted in English.
    italian_voices: list[dict] = []
    seen_ids: set[str] = set()

    for params in (
        dict(accent="italian", language="en", page_size=100),
        dict(language="it", page_size=100),
    ):
        try:
            resp = client.voices.get_shared(**params)
        except Exception as e:
            print(f"  query {params} failed: {e}")
            continue
        for v in resp.voices:
            if v.voice_id in seen_ids:
                continue
            seen_ids.add(v.voice_id)
            italian_voices.append({
                "voice_id": v.voice_id,
                "public_owner_id": getattr(v, "public_owner_id", None),
                "name": v.name,
                "gender": getattr(v, "gender", "?"),
                "age": getattr(v, "age", "?"),
                "accent": getattr(v, "accent", "?"),
                "language": getattr(v, "language", "?"),
                "preview_url": getattr(v, "preview_url", None),
                "use_case": getattr(v, "use_case", None),
            })

    print(f"\nFound {len(italian_voices)} Italian-accent voices:")
    for v in italian_voices[:25]:
        print(f"  {v['voice_id']}  {v['name'][:30]:<30}  ({v['gender']}, {v['age']}, {v['accent']}/{v['language']})")
    if len(italian_voices) > 25:
        print(f"  ... and {len(italian_voices) - 25} more")

    return italian_voices


def generate_samples(
    api_key: str,
    voice_ids: list[str],
    count: int,
    label: str,
    phrase: str,
    model: str = "eleven_v3",
) -> None:
    """Generate `count` total samples spread across `voice_ids`, each with
    rotating phrase variants of `phrase`. Writes 16kHz mono WAV to
    OUTPUT_DIR/<label>/."""
    PHRASE_VARIANTS = build_phrase_variants(phrase)
    try:
        from elevenlabs import ElevenLabs
    except ImportError:
        sys.exit("Install: py -3 -m pip install elevenlabs")

    if not voice_ids:
        sys.exit("ERROR: no voice IDs provided")

    out_dir = OUTPUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    client = ElevenLabs(api_key=api_key)
    samples_per_voice = max(1, count // len(voice_ids))
    total_planned = samples_per_voice * len(voice_ids)

    print(f"\nGenerating ~{total_planned} samples across {len(voice_ids)} voices")
    print(f"Output: {out_dir}")
    print(f"Model:  {model}")
    print()

    saved = 0
    started = time.time()
    for vi, voice_id in enumerate(voice_ids, 1):
        for n in range(samples_per_voice):
            phrase = PHRASE_VARIANTS[n % len(PHRASE_VARIANTS)]
            try:
                # v3 ignores most v2 voice_settings and prefers tag-driven
                # variation; for v2/turbo we still pass the per-call settings
                # to inject randomness.
                kwargs = dict(
                    voice_id=voice_id,
                    text=phrase,
                    model_id=model,
                    output_format="pcm_16000",
                )
                if not model.startswith("eleven_v3"):
                    kwargs["voice_settings"] = {
                        "stability": 0.30 + (random.random() * 0.4),
                        "similarity_boost": 0.5,
                        "style": random.random() * 0.3,
                        "use_speaker_boost": True,
                    }
                audio_iter = client.text_to_speech.convert(**kwargs)
                pcm_bytes = b"".join(audio_iter)
                # Trim leading/trailing silence and re-pad to a uniform 80ms
                # head + 80ms tail. v3 emits inconsistent leading silence
                # (long for whispers, near-zero for shouts); we want every
                # training sample aligned the same way.
                pcm_bytes = _trim_and_pad(pcm_bytes, sample_rate=16000,
                                          head_ms=80, tail_ms=80,
                                          silence_threshold=400)
                wav_path = out_dir / f"{label}_{vi:02d}_{voice_id[:8]}_{n:04d}.wav"
                _write_pcm_wav(wav_path, pcm_bytes, sample_rate=16000)
                saved += 1
                if saved % 25 == 0:
                    elapsed = time.time() - started
                    rate = saved / elapsed if elapsed else 0
                    eta = (total_planned - saved) / rate if rate else 0
                    print(f"  {saved}/{total_planned} ({rate:.1f}/s, ETA {eta:.0f}s)")
            except Exception as e:
                print(f"  ERROR voice={voice_id[:8]} n={n}: {e}")

    print(f"\nDone. {saved}/{total_planned} written to {out_dir}")


def _trim_and_pad(pcm_bytes: bytes, sample_rate: int, head_ms: int,
                  tail_ms: int, silence_threshold: int = 400) -> bytes:
    """Trim leading/trailing silence then pad to fixed head/tail duration.

    Operates on 16-bit signed mono PCM. Silence threshold is the absolute
    sample value below which a frame counts as silence — 400 (~ -38 dBFS)
    handles whisper noise floor without clipping the start of soft consonants.
    Falls back to returning the original bytes if the buffer looks empty."""
    import array
    samples = array.array("h")
    samples.frombytes(pcm_bytes)
    if not samples:
        return pcm_bytes

    # Find first/last non-silent sample (10ms window for stability)
    win = max(1, sample_rate // 100)  # 10ms
    start_i = 0
    for i in range(0, len(samples) - win, win):
        if max(abs(s) for s in samples[i:i + win]) >= silence_threshold:
            start_i = max(0, i - win)  # back off one window for soft attack
            break
    end_i = len(samples)
    for i in range(len(samples) - win, 0, -win):
        if max(abs(s) for s in samples[i:i + win]) >= silence_threshold:
            end_i = min(len(samples), i + 2 * win)  # forward one window
            break
    if end_i <= start_i:
        return pcm_bytes  # nothing detected, leave alone

    trimmed = samples[start_i:end_i]
    head_samples = (head_ms * sample_rate) // 1000
    tail_samples = (tail_ms * sample_rate) // 1000
    pad_head = array.array("h", [0] * head_samples)
    pad_tail = array.array("h", [0] * tail_samples)
    return (pad_head + trimmed + pad_tail).tobytes()


def _write_pcm_wav(path: Path, pcm_bytes: bytes, sample_rate: int = 16000) -> None:
    """Wrap raw PCM in a WAV container."""
    import struct
    n_samples = len(pcm_bytes) // 2  # 16-bit
    byte_rate = sample_rate * 2
    block_align = 2
    data_size = len(pcm_bytes)
    riff_size = 36 + data_size
    header = b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
    fmt = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, block_align, 16)
    data = b"data" + struct.pack("<I", data_size) + pcm_bytes
    path.write_bytes(header + fmt + data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phrase", required=True,
                    help='wake phrase to generate, e.g. "Hey Jeeves"')
    ap.add_argument("--mode", choices=["italian", "cloned", "diverse"],
                    default="diverse",
                    help="sample-generation mode (default: diverse)")
    ap.add_argument("--count", type=int, default=2000,
                    help="total samples to generate (spread across voices)")
    ap.add_argument("--voice-id", help="for --mode cloned: the cloned voice ID")
    ap.add_argument("--list-italian-voices", action="store_true",
                    help="list available Italian-accent voices and exit")
    ap.add_argument("--model", default="eleven_v3",
                    help="ElevenLabs model. v3 = strongest prosody-tag handling. "
                         "multilingual_v2 = good tags + accent. "
                         "turbo_v2_5 = fastest, ignores most bracketed tags.")
    args = ap.parse_args()

    api_key = load_api_key()

    if args.list_italian_voices:
        list_italian_voices(api_key)
        return

    if args.mode == "italian":
        voices = list_italian_voices(api_key)
        if not voices:
            sys.exit("No Italian voices found. Try --mode diverse or check API key.")
        voice_ids = [v["voice_id"] for v in voices[:8]]
        generate_samples(api_key, voice_ids, args.count, "italian", args.phrase, args.model)

    elif args.mode == "cloned":
        if not args.voice_id:
            sys.exit("--voice-id required for cloned mode. Create the clone in ElevenLabs UI first.")
        generate_samples(api_key, [args.voice_id], args.count, "cloned", args.phrase, args.model)

    elif args.mode == "diverse":
        # Hand-picked diverse voices across accents — refresh this list periodically
        diverse_voice_ids = [
            "21m00Tcm4TlvDq8ikWAM",  # Rachel — US female
            "ErXwobaYiN019PkySvjV",  # Antoni — US male
            "MF3mGyEYCl7XYWbV9V6O",  # Elli — US female
            "TxGEqnHWrfWFTfGW9XjX",  # Josh — US male
            "VR6AewLTigWG4xSOukaG",  # Arnold — US male
            "pNInz6obpgDQGcFmaJgB",  # Adam — US male
        ]
        generate_samples(api_key, diverse_voice_ids, args.count, "diverse", args.phrase, args.model)


if __name__ == "__main__":
    main()
