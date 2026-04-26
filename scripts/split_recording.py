#!/usr/bin/env python3
"""
Split one long recording of N×"Hey Harold" utterances into N individual
1-2s WAV files, ready for the training pipeline.

Uses ffmpeg's silencedetect filter to find silence boundaries, then slices
each non-silent region with a small head/tail pad. Resamples to 16kHz mono
16-bit PCM (the format microWakeWord expects).

Usage:
  py -3 split_recording.py path/to/recording.m4a
  py -3 split_recording.py path/to/recording.wav --label alfie
  py -3 split_recording.py path/to/recording.wav --silence-db -35 --min-silence 0.5

Output:
  ./output/real_recordings/<label>/<label>_0001.wav ... _NNNN.wav
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "real_recordings"


def detect_silence(input_path: Path, silence_db: int, min_silence: float) -> list[tuple[float, float]]:
    """Run ffmpeg silencedetect, return list of (utterance_start, utterance_end) in seconds.

    silencedetect emits 'silence_start: T' and 'silence_end: T | silence_duration: D'
    on stderr. We invert: utterances are the gaps BETWEEN silences (plus head before
    first silence and tail after last)."""
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(input_path),
        "-af", f"silencedetect=noise={silence_db}dB:d={min_silence}",
        "-f", "null", "-",
    ]
    print(f"Running: {' '.join(cmd)}\n")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = proc.stderr

    # Total duration
    dur_match = re.search(r"Duration:\s+(\d+):(\d+):(\d+\.\d+)", log)
    if not dur_match:
        sys.exit("Could not determine input duration")
    h, m, s = dur_match.groups()
    total_dur = int(h) * 3600 + int(m) * 60 + float(s)

    silence_starts = [float(m) for m in re.findall(r"silence_start:\s+(-?\d+\.?\d*)", log)]
    silence_ends = [float(m) for m in re.findall(r"silence_end:\s+(-?\d+\.?\d*)", log)]

    # Pair up: utterances live in the gaps between silences
    # Pattern: [voice] silence_start | silence_end [voice] silence_start | silence_end [voice] ...
    utterances = []
    cursor = 0.0
    for s_start, s_end in zip(silence_starts, silence_ends):
        if s_start > cursor + 0.05:  # there's a voiced segment before this silence
            utterances.append((max(0.0, cursor), s_start))
        cursor = s_end
    # Tail after last silence
    if cursor < total_dur - 0.05:
        utterances.append((cursor, total_dur))

    return utterances


def slice_utterances(
    input_path: Path,
    utterances: list[tuple[float, float]],
    out_dir: Path,
    label: str,
    pad: float = 0.10,
    target_min: float = 0.6,
    target_max: float = 2.0,
) -> int:
    """Slice each utterance to its own 16kHz mono WAV, padded by `pad` seconds
    each side. Drops slices outside target_min..target_max length (likely noise
    bursts or run-on phrases)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    skipped = 0
    for i, (start, end) in enumerate(utterances, 1):
        dur = end - start
        if not (target_min <= dur <= target_max):
            skipped += 1
            print(f"  skip #{i:03d}  {dur:.2f}s  (outside {target_min}-{target_max}s window)")
            continue
        clip_start = max(0.0, start - pad)
        clip_dur = (end - start) + 2 * pad
        out_path = out_dir / f"{label}_{saved + 1:04d}.wav"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{clip_start:.3f}", "-t", f"{clip_dur:.3f}",
            "-i", str(input_path),
            "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR slice #{i}: {result.stderr.strip()}")
            continue
        saved += 1
        if saved % 10 == 0:
            print(f"  {saved} saved...")
    print(f"\nDone. {saved} clips saved to {out_dir}, {skipped} skipped.")
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="long recording (any format ffmpeg reads)")
    ap.add_argument("--label", default="alfie", help="output filename prefix + subdir")
    ap.add_argument("--silence-db", type=int, default=-35,
                    help="silence threshold in dB (default -35; quieter rooms can use -40)")
    ap.add_argument("--min-silence", type=float, default=0.5,
                    help="minimum silence duration to count as a gap (default 0.5s)")
    ap.add_argument("--pad", type=float, default=0.10,
                    help="head/tail padding around each utterance (default 0.10s)")
    ap.add_argument("--min-len", type=float, default=0.6,
                    help="drop slices shorter than this (default 0.6s)")
    ap.add_argument("--max-len", type=float, default=2.0,
                    help="drop slices longer than this (default 2.0s)")
    ap.add_argument("--dry-run", action="store_true",
                    help="just detect + report, don't slice")
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        sys.exit(f"ERROR: {input_path} not found")

    utterances = detect_silence(input_path, args.silence_db, args.min_silence)
    print(f"Detected {len(utterances)} candidate utterances:")
    for i, (s, e) in enumerate(utterances[:5], 1):
        print(f"  #{i:03d}  {s:6.2f}s -> {e:6.2f}s  ({e - s:.2f}s)")
    if len(utterances) > 5:
        print(f"  ... and {len(utterances) - 5} more")

    if args.dry_run:
        return

    out_dir = OUTPUT_DIR / args.label
    slice_utterances(input_path, utterances, out_dir,
                     args.label, args.pad, args.min_len, args.max_len)


if __name__ == "__main__":
    main()
