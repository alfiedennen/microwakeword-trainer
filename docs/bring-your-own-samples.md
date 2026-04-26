# Bring Your Own Samples (BYOS) workflow

For when you want **higher quality** than what Piper-in-Colab gives you. You
do the sample-prep locally, upload a zip, run the notebook in `bundle` mode.

## Why bother

`generate` mode (Piper in Colab) uses 904 US-English LibriTTS-R speakers.
That's solid English coverage but:

- **Only US English accent** — no British, Indian, Italian, Spanish, French
  speakers reading English. If your household has accent diversity, the model
  won't generalise to non-US accents well.
- **Synthetic-only positives** — no real recordings of you. Real I2S-mic audio
  has reverb/noise/AGC profiles that no TTS replicates.
- **No targeted confusables** — the default list is generic; the most
  effective confusables are wake-word-specific.

`bundle` mode lets you address all three.

## Bundle layout

A zip file containing three subdirs:

```
generated_samples/        positive samples (.wav, 16 kHz mono)
real_recordings/          (optional) real mic recordings of the wake word
confusable_negatives/     (optional) hard-negative samples
```

Drop into Drive at `<DRIVE_FOLDER>/data_bundle.zip`, set `MODE = "bundle"`
in the notebook, Run All.

## Building the bundle

### Real recordings (high-impact, easiest gain)

Record one long .m4a or .wav of yourself saying the wake word ~60-100 times
with ~1.5s pauses between each. Vary distance, volume, prosody.

Then:

```bash
py -3 scripts/split_recording.py path/to/your_recording.m4a --label your_name
```

This silence-detects the gaps + slices into individual normalized 16 kHz mono
WAVs in `real_recordings/your_name/`.

If you have multiple speakers (e.g. partner, kids), do separate recordings per
person — the trainer balances them.

### Accent-diverse TTS (ElevenLabs)

ElevenLabs has 5000+ voices including native non-English speakers reading
English with their natural accent. Far better than Piper for accent coverage.

```bash
export ELEVENLABS_API_KEY=sk_xxx
py -3 scripts/elevenlabs_generate.py --phrase "Hey Jeeves" --mode italian --count 2000
py -3 scripts/elevenlabs_generate.py --phrase "Hey Jeeves" --mode diverse --count 1000
```

Cost: a short wake phrase × 22 prosody templates × 2000 samples ≈ 60-80k
chars. Within ElevenLabs Creator-tier (£11/mo, 100k chars/mo). Don't try this
on Free tier — only 10k chars/mo.

If you want a clone of a specific speaker: create the voice clone in the
ElevenLabs UI from a 1-minute recording, get the voice ID, then:

```bash
py -3 scripts/elevenlabs_generate.py --phrase "Hey Jeeves" --mode cloned --voice-id <ID> --count 1000
```

### Bulk Piper locally (optional, alternative to in-Colab generation)

If you want to iterate faster without burning Colab compute:

```bash
py -3 scripts/piper_generate.py --count 3000
```

Generates ~3000 US-English samples across 904 LibriTTS-R speakers.
Takes ~10 min on a 6-core CPU.

### Confusables (the most underrated knob)

Make a list of phonetic neighbors, the bare name without "Hey" prefix, and
other common false-fires for your wake word. Generate samples of each via
`elevenlabs_generate.py` (with `--phrase "ey arold"` for example), drop them
into `confusable_negatives/`.

Or use the Piper-based `generate_confusables.py` (TODO: not in repo yet —
see Hey Harold session for the pattern).

## Bundling

Once you've got `generated_samples/`, `real_recordings/`, `confusable_negatives/`
populated:

```bash
py -3 scripts/bundle_for_colab.py
```

(TODO: this script is in the Harold-Road project; will be ported to this repo.
For now: just `cd` to a parent dir of the three folders and `zip -r data_bundle.zip
generated_samples/ real_recordings/ confusable_negatives/`.)

Upload the zip to Drive at `<DRIVE_FOLDER>/data_bundle.zip`.

## Tips

- **Don't oversample any one voice.** The trainer balances by source dir, so
  if you have 100 real recordings and 10000 TTS, the real ones are weighted
  heavily — that's good.
- **Match sample format.** All WAVs should be 16 kHz mono 16-bit PCM. The
  helper scripts produce this; check your own with `ffprobe`.
- **Trim silence to ~80 ms head + 80 ms tail.** Sample-aligned start/end
  helps the model learn consistent timing. The helper scripts do this.
- **Real recordings beat synthetic 10:1 for the household's primary user.**
  Even 50 real samples shift performance noticeably.
