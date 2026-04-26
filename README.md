# microWakeWord Community Trainer — Custom wake words for Home Assistant + ESPHome

Train your own custom wake word for **Home Assistant Voice**, **ESPHome
`micro_wake_word`**, **M5Stack Atom Echo**, **Nabu Casa Voice PE**, and any
other ESPHome-based voice satellite. One Google Colab notebook, two modes
("Piper-generates-everything" or "bring your own samples"), no kernel-restart
dance, no local GPU required.

**Train**, **deploy**, and **iterate** custom wake words for your **HA voice
assistant** without paying for cloud STT or surrendering your audio to
Google/Amazon. Privacy-first home automation that triggers only on YOUR phrase.

If you've struggled with the official microWakeWord training notebook
("ModuleNotFoundError: microwakeword.audio", "Failed to allocate tensors",
"silent training failure", "OOM on T4"), this repo has the working fixes
baked in.

**Tags**: home-assistant, esphome, micro-wake-word, m5stack-atom-echo,
voice-pe, nabu-casa, custom-wake-word, wake-word-detection, on-device-ml,
tensorflow-lite, tflite-micro, esp32, voice-assistant, smart-home, openwakeword,
keyword-spotting, openhomefoundation

This repo is a community-friendly wrapper around [kahrendt/microWakeWord](https://github.com/kahrendt/microWakeWord)
with every gotcha from the production deployment of "Hey Harold" already
patched. See [examples/hey_harold/](examples/hey_harold/) for the working model
running on 5× M5Stack Atom Echo voice satellites in a Home Assistant household.

## Quickstart (5 minutes)

1. Open the notebook in Colab:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GH_USERNAME/microwakeword-trainer/blob/main/notebooks/microWakeWord_train_any_wakeword.ipynb)

2. **Runtime → Change runtime type → A100 GPU + High-RAM** *(non-negotiable — see Gotchas)*

3. Edit the `CONFIGURE YOUR WAKE WORD HERE` cell:
   ```python
   WAKE_WORD = "Hey Jeeves"
   OUTPUT_NAME = "hey_jeeves"
   WAKE_WORD_IPA_US = "hˈeɪ dʒˈivz"   # see "Finding the IPA" below
   MODE = "generate"                  # Piper makes samples in Colab
   ```

4. **Runtime → Run all**, walk away ~45 minutes.

5. Find `hey_jeeves.tflite` + `hey_jeeves.json` in your Drive folder.
   Drop them into your ESPHome config and reflash.

## Two modes

### `generate` (easiest)

Piper TTS generates ~30,000 positive samples + your confusables inside Colab.
You provide an IPA pronunciation; the notebook does everything else.

**Best for:** trying a new wake word fast, common English-language words.

### `bundle` (highest quality)

You build your own training bundle locally and upload it. Use this when you want:
- **Real recordings** of yourself / household members saying the wake word
- **Accent-matched samples** (e.g. ElevenLabs voices in your speaker's native accent)
- **Hand-curated confusables** (phonetic neighbors, common false-fires)

The repo includes helper scripts under `scripts/` to build a bundle:
- `split_recording.py` — slice one long .m4a/.wav of N×wakeword utterances into individual sample WAVs
- `elevenlabs_generate.py` — accent-diverse positive samples via ElevenLabs (Italian-accent, voice clones, etc)
- `piper_generate.py` — bulk US-English samples locally (no Colab needed)

Bundle layout (zipped as `data_bundle.zip` in your Drive folder):

```
generated_samples/        positive samples (.wav, 16 kHz mono)
real_recordings/          (optional) real mic recordings
confusable_negatives/     (optional) hard negatives — phonetic near-misses
```

## Finding the IPA pronunciation of your wake word

Easiest:
```bash
sudo apt install espeak-ng
espeak-ng -q --ipa "Hey Jeeves"
# → hˈeɪ dʒˈivz
```

Or use the [International Phonetic Association chart](https://www.internationalphoneticassociation.org/IPA-chart)
or [ipa-reader.com](https://ipa-reader.com).

For wake words with non-English elements (proper nouns, foreign words), you may
need to write the IPA manually. The `--phoneme-input` flag in Piper takes IPA
verbatim, so as long as your IPA is correct it'll work.

## Manifest tuning (likely needed after first deploy)

The defaults work for **Hey Harold** specifically. Your model's confidence
distribution will differ. After your first deploy:

| Symptom | Fix |
|---|---|
| Doesn't fire on the wake word | Lower `probability_cutoff` (try 0.7, 0.6) |
| Fires on too many things | Raise `probability_cutoff` (try 0.92, 0.95) |
| LED fires but no STT response | Confirm `sliding_window_size: 5` (faster fire) |
| `Failed to allocate tensors` log | Raise `tensor_arena_size` (try 50000, 80000) |

Edit your `<output_name>.json` manifest, no retraining needed.

## Confusables — the unsung hero

The biggest false-trigger reduction comes from training with deliberate
hard-negative samples (phrases that sound near your wake word but should NOT
fire). Edit the `CONFUSABLE_PHRASES` list in the config cell to include:

- **Phonetic neighbors of your wake word** — for "Hey Harold": Hey Howard, Hey
  Harvey, Hey Gerald
- **The bare name without the prefix** — for "Hey Harold": "Harold" alone
- **Other assistant wake words** — Hey Siri, Okay Google, Okay Nabu (so it
  doesn't steal them)
- **Phonetic mash-ups** — for "Hey Harold": "hairy old"
- **Generic "hey X"** — Hey there, Hey you, Hey y'all

Each gets ~1000 samples generated with prosody variation. They're weighted
heavily as negatives during training.

## Gotchas baked into the notebook (don't undo)

These caused real failures during the development of this notebook. The fixes
are all applied; documenting them so you don't re-introduce them.

1. **A100 + High-RAM runtime is required.** T4 OOMs during validation
   (allocation > free system RAM). Don't even try.
2. **Don't `pip install -e ./microWakeWord`** — needs kernel restart, breaks
   Run All. The notebook uses `sys.path.insert` instead.
3. **Don't `pip install ./microWakeWord` either** — kahrendt's setup.py has no
   `find_packages()` declaration, so non-editable install only grabs top-level
   files (skips the `audio/` subpackage). Same `sys.path` workaround.
4. **`train.py` calls `.numpy()` on numpy arrays under newer TF.** The notebook
   monkey-patches the file with a `hasattr()` guard.
5. **Subprocess training needs `PYTHONPATH=/content/microWakeWord`** — the
   `sys.path` from Jupyter doesn't propagate to subprocesses.
6. **`voice_assistant` has no audio lookback buffer.** When MWW fires, audio
   capture starts from that moment forward — pre-fire audio is gone. So the
   model needs to fire fast (low `sliding_window_size`) so the user's command
   is captured in full.

## Hardware tested

- **M5Stack Atom Echo** (ESP32, 4 MB flash, 320 KB SRAM) — proven working,
  5 units running in production household
- Other ESPHome `micro_wake_word`-capable devices likely fine — please open
  an issue with results:
  - Nabu Casa **Voice Preview Edition** (Voice PE)
  - **ESP32-S3 Box / Box-3** (Espressif)
  - Generic **ESP32 + I2S microphone** boards
  - **Home Assistant Voice** Beta hardware

## Why train your own wake word for Home Assistant?

The built-in wake words (`okay_nabu`, `hey_jarvis`, `alexa`) work well, but a
**custom wake word** lets you:

- **Stop the cross-talk** with Alexa/Google devices in the same room
- **Reflect your household's identity** — name your assistant after a person,
  pet, or character
- **Avoid colliding** with the assistants on your phones (Siri, Google
  Assistant)
- **Train on your accent / household voices** for higher recall, especially
  for non-US-English speakers
- **Truly local** — wake-word detection happens entirely on the ESP32, no
  audio leaves the device until *after* the wake word fires

## Acknowledgements

- [kahrendt/microWakeWord](https://github.com/kahrendt/microWakeWord) — the
  underlying training framework
- [malonestar/microWakeWord-data](https://github.com/malonestar/microWakeWord-data)
  — proven-working notebook patterns I lifted heavily from
- [esphome/micro-wake-word-models](https://github.com/esphome/micro-wake-word-models)
  — reference manifest values for okay_nabu / hey_jarvis / etc
- [rhasspy/piper-sample-generator](https://github.com/rhasspy/piper-sample-generator)
  — the TTS-based positive generation
- [Open Home Foundation](https://www.openhomefoundation.org/) — for funding
  the wake-word ecosystem

## Contributing

Working models, training tips, manifest values for your wake word — all welcome.
Open a PR to `examples/<your_wake_word>/` with the `.tflite`, `.json`, and a
short README of what worked.

## License

MIT — see [LICENSE](LICENSE).
