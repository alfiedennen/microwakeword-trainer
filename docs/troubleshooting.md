# Troubleshooting

## Notebook fails

### `ModuleNotFoundError: No module named 'microwakeword'` or `microwakeword.audio`

The install cell didn't run, or it ran but the kernel needs restarting.
**Don't restart the kernel** — instead, the notebook uses `sys.path.insert`
which avoids the need. If you see this error, scroll up to the install cell
and re-run just that one. You should see `OK: microwakeword.audio.* imports
clean` at the end.

### `Failed to allocate tensors for the streaming model`

This is a runtime error on the **device side**, not in Colab. Your
`tensor_arena_size` in the manifest is too small for your model. Try `50000`
(the default) or `80000`. Edit the `.json` next to your `.tflite` — no
retraining needed.

### Training cell exits with non-zero code

The notebook hard-fails on training errors. The traceback above the assertion
tells you why. Common ones:

- **OOM**: switch to A100 + High-RAM runtime (mandatory anyway)
- **CUDA driver version mismatch**: usually cleared by Disconnect-and-delete
  runtime + Run All again
- **`AttributeError: 'numpy.ndarray' object has no attribute 'numpy'`**: the
  install-cell patch didn't apply. Re-run the install cell, then training.

### Manifest cell says "Training did not produce a model"

Means training finished but produced no .tflite. Check the training cell
output — there's a real error there that wasn't caught. Most common: the
model architecture didn't converge enough to export.

### Push-to-Drive cell errors

Drive auth probably timed out. Re-run from the Drive mount cell down — it'll
re-prompt for auth.

## Wake word doesn't fire

### LED never turns blue when you say the wake word

Probably one of:

1. **`probability_cutoff` too high** — your model's confidence on real
   utterances doesn't reach the cutoff. Try lowering: `0.85 → 0.7 → 0.6`.
   Edit the `.json` manifest, reflash, retest.
2. **`tensor_arena_size` too small** — model failed to load on device. Check
   ESPHome logs for `Failed to allocate tensors`. Bump to 50000 or 80000.
3. **Wrong wake-word file path** — the `model:` field in the YAML doesn't
   resolve. Test with `esphome compile <yaml>` and check for "model not
   found" errors.

### LED fires but no STT response

Logs show `stt-no-text-recognized`. The wake word fired LATE — voice_assistant
started capturing audio AFTER you finished speaking. Two fixes:

1. **`sliding_window_size: 5`** is okay_nabu's default and the right choice
   for fast fire. Don't go higher.
2. **Train v2 with samples that have less leading silence** — your model is
   learning to fire late. The community notebook trims to 80 ms head +
   80 ms tail by default.

### Fires on too many things (false positives)

1. **`probability_cutoff` too low** — raise from 0.85 to 0.92 or 0.95.
2. **Need more confusables** — what's it firing on? Add those phrases to
   `CONFUSABLE_PHRASES` and retrain.

### Fires on the bare name (e.g. "Harold" without "Hey")

Add the bare name to your confusables. This is the #1 most common gap in
custom wake words.

## Pipeline works for some queries but not others

### Device commands work ("turn on light"), info queries don't ("what time")

That's a Home Assistant pipeline issue, not a wake word issue. Different
conversation agents handle these differently. If you're using Harold or a
similar custom agent, check whether it's silencing query responses. See
[harold_speech_slots_bug](https://github.com/...) for one such example.

## Hardware-specific

### M5Stack Atom Echo

- The internal speaker is for TTS responses by default; mute via
  `volume_multiplier: 0.001` in the `voice_assistant:` block.
- Flash via USB-C the first time, OTA after.
- ESP-IDF compile is slow (~5 min). Cache builds locally; PlatformIO rejects
  paths with whitespace, so keep your build dir at e.g. `C:\esp_flash_tmp\`.

### Voice PE / other devices

Untested by us — please open an issue with what worked / didn't.
