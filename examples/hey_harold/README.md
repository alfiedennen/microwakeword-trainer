# Hey Harold — proof-of-deployment

This is the model that was used to develop and validate the trainer in this repo.
Trained 2026-04-26 on **4774 positive samples + 1900 hard negatives**, deployed
to 5× M5Stack Atom Echo voice satellites running Home Assistant.

## What's here

- `hey_harold.tflite` (60 KB) — the trained model
- `hey_harold.json` — ESPHome `micro_wake_word` v2 manifest with the
  manifest values that work in production

You can drop these straight into your ESPHome config to test that
"Hey Harold" wake-word detection works on your hardware before training your
own.

## Training data composition

| Source | Count | Notes |
|---|---|---|
| Real recordings (Alfie, the household's primary user) | 62 | One long m4a, ffmpeg `silencedetect`-sliced |
| ElevenLabs Italian-accent voices (`eleven_v3` + prosody tags) | 2000 | For Elena, the household's Italian-native speaker |
| Piper US-English (libritts-r 904 speakers) | 2712 | Bulk variety |
| Piper US confusables (Howard / Harvey / Gerald / "hey Siri" etc) | 1500 | Hard negatives |
| ElevenLabs Italian confusables (ey arold, aroldo, ehi) | 400 | Italian-specific near-misses |

## Manifest values (deployed)

```json
{
  "probability_cutoff": 0.85,
  "sliding_window_size": 5,
  "tensor_arena_size": 50000
}
```

These differ from the as-trained defaults (which were `0.5`, `5`, `30000`) —
they were tuned during deployment based on what actually worked end-to-end on
the M5Stack Atom Echo. See the manifest tuning section in the main
[README](../../README.md) for the symptom→fix table.

## Test it on your device

Add to your ESPHome YAML:

```yaml
micro_wake_word:
  models:
    - model: hey_harold.json   # path relative to esphome config dir
  on_wake_word_detected:
    - voice_assistant.start:
```

Place `hey_harold.tflite` and `hey_harold.json` next to the YAML, or in a
subdirectory referenced by the path.

Recompile + flash. Test with: "Hey Harold" — LED should fire, voice pipeline
should activate.

## Known characteristics

- **Doesn't false-trigger on**: "Hey Howard", "Hey Harvey", "Hey Gerald"
  (trained as confusables)
- **Recognises both**: native English ("Hey Harold") and Italian-accented
  English ("ey arold")
- **Doesn't fire on**: bare "Harold", "Hey Siri", "Okay Nabu" (all confusables)
- **One known weakness**: needs a longer-than-Okay-Nabu pause between wake
  word and command, especially for long utterances. The model's confidence
  curve is less peaked than the heavily-tuned built-in models. v2 retraining
  with more leading silence in positive samples would address this.
