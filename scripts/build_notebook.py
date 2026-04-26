#!/usr/bin/env python3
"""
Build microWakeWord_train_any_wakeword.ipynb — a dual-mode community-friendly
training notebook for custom wake words.

What this notebook does:
  - Two modes selectable from the top config cell:
    * MODE = "generate" → Piper TTS generates training samples in Colab
                          using an IPA pronunciation. Easiest path.
    * MODE = "bundle"   → User-supplied zip in Drive with their own samples
                          (real recordings, custom TTS, accent-matched).
  - All bug fixes baked in (kahrendt setup.py, train.py .numpy(),
    PYTHONPATH subprocess, hardened asserts, A100+High-RAM requirement)
  - Manifest values pre-tuned to known-working defaults (cutoff 0.85,
    sliding_window 5, tensor_arena 50000)
  - One-click "Run all" workflow

Output: wakeword/notebooks/microWakeWord_train_any_wakeword.ipynb
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"
DEST = NOTEBOOK_DIR / "microWakeWord_train_any_wakeword.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


CELLS = []

# ─── Title + intro ────────────────────────────────────────────────────────
CELLS.append(md("""# microWakeWord — Train any wake word

A single-notebook trainer for custom wake words on ESPHome `micro_wake_word` devices
(M5Stack Atom Echo, Voice PE, etc).

Two modes:
- **`generate`** — Piper TTS generates ~30k positive samples + your confusables in Colab.
  Easiest path. Requires an IPA pronunciation of your wake word.
- **`bundle`** — You upload a zip with your own samples (real recordings, ElevenLabs
  voices, accent-matched TTS). Higher quality but you do the prep work.

## Required runtime
**Runtime → Change runtime type → A100 GPU + High-RAM**.
T4 OOMs during validation. A100 + High-RAM gives 40 GB VRAM + 85 GB system RAM.

## What you get
A `<output_name>.tflite` (~60 KB) + companion `.json` manifest, ready to drop into
your ESPHome config under `micro_wake_word: models:`.

## Workflow
1. Edit the **CONFIGURE HERE** cell below for your wake word
2. **Runtime → Run all**, walk away ~45 minutes
3. Find `<output_name>.tflite` + `.json` in your Drive folder when it's done
4. Test on hardware — likely needs manifest tuning (cutoff, sliding_window) for
   your specific model. See the deployment notes at the end.

Built from working production deployment of "Hey Harold" — all known
upstream bugs are patched in this notebook.
"""))

# ─── USER CONFIG cell ────────────────────────────────────────────────────
CELLS.append(code('''# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                    CONFIGURE YOUR WAKE WORD HERE                       ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# ─── Wake word identity ───
WAKE_WORD = "Hey Harold"           # Human-readable name (shown in HA)
OUTPUT_NAME = "hey_harold"         # Filename (no spaces, lowercase)
AUTHOR = "your_name"               # Goes in the manifest
AUTHOR_WEBSITE = "https://github.com/yourname"

# ─── Mode ───
MODE = "generate"   # "generate" (Piper makes samples) | "bundle" (you uploaded a zip)

# ─── Drive folder (created if missing) ───
DRIVE_FOLDER = "wakeword_training_hey_harold"

# ─── If MODE == "generate" ───
# Look up IPA for your wake word: https://www.internationalphoneticassociation.org/IPA-chart
# or use eSpeak: `espeak-ng -q --ipa "Hey Harold"` on Linux
WAKE_WORD_IPA_US = "hˈeɪ hˈærəld"   # US English pronunciation
WAKE_WORD_IPA_UK = "hˈeɪ hˈɛrəld"   # UK English (set to None to skip second pass)
SAMPLES_US = 30000                  # ~12 min on T4, ~7 min on A100
SAMPLES_UK = 15000                  # 0 to skip UK pass

# Confusable phrases — should NOT trigger your wake word.
# Include: phonetic neighbors, the bare name without prefix, common
# false-trigger phrases like other assistant names.
CONFUSABLE_PHRASES = [
    # General "hey X" near-misses
    "hey there", "hey you", "hey y'all", "hey now",
    # Other assistant wake words (must not steal yours)
    "hey siri", "hey google", "okay google", "hey alexa", "okay nabu",
    # WAKE-WORD SPECIFIC — replace these for your word!
    "hey howard", "hey harvey", "hey gerald", "hey carol",  # H-name neighbors
    "harold", "the herald",                                  # bare name + similar
    "hairy old", "hello harold",                              # phonetic mash-ups
]
SAMPLES_PER_CONFUSABLE = 1000   # 500 = light, 1000 = recommended, 2000 = max

# ─── If MODE == "bundle" ───
# Expected: <DRIVE_FOLDER>/data_bundle.zip with this layout:
#   generated_samples/        positive samples (.wav, 16 kHz mono)
#   real_recordings/          (optional) real mic recordings
#   confusable_negatives/     (optional) hard negative samples
BUNDLE_NAME = "data_bundle.zip"

# ─── Manifest tuning (works well as defaults; tune after on-device testing) ───
PROBABILITY_CUTOFF = 0.85          # 0.5 = too lenient, 0.97 = okay_nabu strict
SLIDING_WINDOW_SIZE = 5            # 5 = okay_nabu default; higher = slower fire
TENSOR_ARENA_SIZE = 50000          # 30000 works, 50000 has more headroom

# ─── Languages this model is trained for (manifest metadata) ───
TRAINED_LANGUAGES = ["en"]         # add "it", "es" etc if you have multi-accent samples

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                           END OF CONFIG                                ║
# ╚═══════════════════════════════════════════════════════════════════════╝
print(f"Training '{WAKE_WORD}' as {OUTPUT_NAME} in mode={MODE!r}")
print(f"Output → /content/drive/MyDrive/{DRIVE_FOLDER}/{OUTPUT_NAME}.tflite")
'''))

# ─── Drive mount ─────────────────────────────────────────────────────────
CELLS.append(code('''# === Mount Drive ===
from google.colab import drive
import os
drive.mount('/content/drive')

DRIVE_DIR = f'/content/drive/MyDrive/{DRIVE_FOLDER}'
os.makedirs(DRIVE_DIR, exist_ok=True)
print(f'Drive folder: {DRIVE_DIR}')
if MODE == 'bundle':
    BUNDLE_PATH = f'{DRIVE_DIR}/{BUNDLE_NAME}'
    assert os.path.exists(BUNDLE_PATH), (
        f'MODE=bundle but {BUNDLE_PATH} does not exist. Upload your data zip there.')
    print(f'Found bundle: {os.path.getsize(BUNDLE_PATH)/1024/1024:.1f} MB')
'''))

# ─── Install microWakeWord (the patched cell from our session) ───────────
CELLS.append(code('''# === Install microWakeWord (kernel-restart-free) ===
# Workarounds for two upstream bugs:
#  1. kahrendt/microWakeWord setup.py has no find_packages() — non-editable
#     install skips the audio/ subpackage. Editable install needs kernel
#     restart, breaks Run All. Fix: install deps + sys.path.insert().
#  2. train.py calls .numpy() on values that newer TF returns as numpy
#     arrays already. Patch with hasattr() guard.
import os, sys, subprocess, importlib, re

DEPS = [
    'audiomentations', 'audio_metadata', 'datasets', 'mmap_ninja', 'numpy',
    'pymicro-features', 'pyyaml', 'tensorflow>=2.16', 'webrtcvad-wheels',
    'ai-edge-litert',
    'git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f',
]
print('Installing dependencies...')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + DEPS, check=True)

if not os.path.exists('microWakeWord'):
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/kahrendt/microWakeWord'], check=True)

MWW_DIR = '/content/microWakeWord'
if MWW_DIR not in sys.path:
    sys.path.insert(0, MWW_DIR)
importlib.invalidate_caches()

fp = '/content/microWakeWord/microwakeword/train.py'
src = open(fp).read()
patched = re.sub(
    r'(\\b[a-zA-Z_]+\\["[a-z]+"\\])\\.numpy\\(\\)',
    r'(\\1.numpy() if hasattr(\\1, "numpy") else \\1)',
    src
)
n = patched.count('hasattr') - src.count('hasattr')
if n > 0:
    open(fp, 'w').write(patched)
    print(f'Patched {n} .numpy() calls in train.py')

import microwakeword
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration
print('OK: microwakeword.audio.* imports clean')
'''))

# ─── DATA: bundle mode ───────────────────────────────────────────────────
CELLS.append(code('''# === Data preparation ===
# Mode-aware: either unzip user's bundle, or generate samples inline via Piper.
import os, zipfile

os.chdir('/content')

if MODE == 'bundle':
    print(f'Extracting {BUNDLE_PATH}...')
    with zipfile.ZipFile(BUNDLE_PATH, 'r') as zf:
        zf.extractall('/content')
    for d in ['generated_samples', 'real_recordings', 'confusable_negatives']:
        p = f'/content/{d}'
        if os.path.exists(p):
            n = sum(1 for _ in os.scandir(p) if _.name.endswith('.wav'))
            print(f'  {d}: {n} WAVs')
        else:
            print(f'  {d}: MISSING (will train without)')

elif MODE == 'generate':
    # Defer to the Piper sample-gen cells below
    print('MODE=generate — Piper sample-gen cells will produce samples')

else:
    raise ValueError(f'Unknown MODE: {MODE!r}. Use "bundle" or "generate".')
'''))

# ─── Piper install (only used in generate mode) ───────────────────────────
CELLS.append(code('''# === Piper sample generator install (skipped if MODE=bundle) ===
if MODE == 'generate':
    import glob, os, shutil, subprocess, sys, urllib.request
    PIPER_REPO_DIR = '/content/piper'
    PIPER_SAMPLE_GENERATOR_DIR = '/content/piper-sample-generator'

    subprocess.run(['apt-get', '-qq', 'install', '-y', 'espeak-ng'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade',
                    'pip', 'setuptools', 'wheel', 'cython'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade',
                    'piper-tts', 'piper-sample-generator'], check=True)

    if not os.path.exists(PIPER_REPO_DIR):
        subprocess.run(['git', 'clone', '--depth', '1',
                        'https://github.com/rhasspy/piper', PIPER_REPO_DIR], check=True)
    if not os.path.exists(PIPER_SAMPLE_GENERATOR_DIR):
        subprocess.run(['git', 'clone', '--depth', '1',
                        'https://github.com/rhasspy/piper-sample-generator',
                        PIPER_SAMPLE_GENERATOR_DIR], check=True)

    PIPER_PYTHON_DIR = f'{PIPER_REPO_DIR}/src/python'
    MA_DIR = f'{PIPER_PYTHON_DIR}/piper_train/vits/monotonic_align'
    MA_IMPORT_DIR = f'{MA_DIR}/monotonic_align'
    MA_BUILD_DIR = f'{MA_DIR}/piper_train/vits/monotonic_align'

    shutil.rmtree(f'{PIPER_PYTHON_DIR}/build', ignore_errors=True)
    shutil.rmtree(MA_IMPORT_DIR, ignore_errors=True)
    shutil.rmtree(f'{MA_DIR}/piper_train', ignore_errors=True)
    os.makedirs(MA_IMPORT_DIR, exist_ok=True)
    os.makedirs(MA_BUILD_DIR, exist_ok=True)
    open(f'{MA_IMPORT_DIR}/__init__.py', 'a').close()
    subprocess.run(f'cd {MA_DIR} && {sys.executable} setup.py build_ext --inplace',
                   shell=True, check=True)
    built = next(iter(glob.glob(f'{MA_BUILD_DIR}/core.*')), None)
    assert built, 'monotonic_align core extension build failed'
    shutil.copy2(built, MA_IMPORT_DIR)

    for path in (PIPER_PYTHON_DIR, PIPER_SAMPLE_GENERATOR_DIR):
        if path not in sys.path:
            sys.path.insert(0, path)

    MODEL_PATH = 'models/en_US-libritts_r-medium.pt'
    MODEL_CONFIG_PATH = f'{MODEL_PATH}.json'
    os.makedirs('models', exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print('Downloading libritts_r model (~75 MB)...')
        urllib.request.urlretrieve(
            'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt',
            MODEL_PATH)
        urllib.request.urlretrieve(
            'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json',
            MODEL_CONFIG_PATH)
    print('Piper ready')
else:
    print('Skipped (MODE != generate)')
'''))

# ─── Generate positives (US + optional UK) ────────────────────────────────
CELLS.append(code('''# === Generate positive samples (US English, optionally UK) ===
if MODE == 'generate':
    import os, subprocess, sys
    PIPER_SAMPLE_GENERATOR_DIR = '/content/piper-sample-generator'
    MODEL_PATH = 'models/en_US-libritts_r-medium.pt'
    PIPER_BATCH = 256  # drop to 128 on T4 if CUDA OOM

    def run_piper(target_word, max_samples, output_dir):
        cmd = [sys.executable, f'{PIPER_SAMPLE_GENERATOR_DIR}/generate_samples.py',
               target_word, '--phoneme-input', '--model', MODEL_PATH,
               '--max-samples', str(max_samples),
               '--batch-size', str(PIPER_BATCH),
               '--noise-scales', '0.5', '--noise-scale-ws', '0.6',
               '--output-dir', output_dir]
        subprocess.run(cmd, check=True)

    os.makedirs('generated_samples', exist_ok=True)

    print(f'Generating {SAMPLES_US} US samples for {WAKE_WORD_IPA_US!r}...')
    run_piper(WAKE_WORD_IPA_US, SAMPLES_US, 'generated_samples')

    if WAKE_WORD_IPA_UK and SAMPLES_UK > 0:
        print(f'Generating {SAMPLES_UK} UK samples for {WAKE_WORD_IPA_UK!r}...')
        run_piper(WAKE_WORD_IPA_UK, SAMPLES_UK, 'generated_samples')

    n = sum(1 for f in os.listdir('generated_samples') if f.endswith('.wav'))
    print(f'Total positive samples: {n}')
else:
    print('Skipped (MODE != generate)')
'''))

# ─── Generate confusable negatives ────────────────────────────────────────
CELLS.append(code('''# === Generate confusable negatives ===
if MODE == 'generate':
    import os, subprocess, sys
    from pathlib import Path
    PIPER_SAMPLE_GENERATOR_DIR = '/content/piper-sample-generator'
    MODEL_PATH = 'models/en_US-libritts_r-medium.pt'

    os.makedirs('confusable_negatives', exist_ok=True)
    for phrase in CONFUSABLE_PHRASES:
        safe = phrase.replace(' ', '_').replace(',', '').replace("'", '')
        existing = len(list(Path('confusable_negatives').glob(f'{safe}_*.wav')))
        if existing >= SAMPLES_PER_CONFUSABLE:
            print(f'  {phrase!r}: {existing} already, skip')
            continue
        tmp = f'/tmp/confusable_{safe}'
        os.makedirs(tmp, exist_ok=True)
        print(f'  generating {SAMPLES_PER_CONFUSABLE} for {phrase!r}...')
        subprocess.run([sys.executable, f'{PIPER_SAMPLE_GENERATOR_DIR}/generate_samples.py',
                        phrase, '--model', MODEL_PATH,
                        '--max-samples', str(SAMPLES_PER_CONFUSABLE),
                        '--batch-size', '256',
                        '--noise-scales', '0.5', '--noise-scale-ws', '0.6',
                        '--output-dir', tmp], check=True)
        # Move + rename with safe prefix
        for f in os.listdir(tmp):
            if f.endswith('.wav'):
                os.rename(f'{tmp}/{f}', f'confusable_negatives/{safe}_{f}')

    n = sum(1 for f in os.listdir('confusable_negatives') if f.endswith('.wav'))
    print(f'Total confusable negatives: {n}')
else:
    print('Skipped (MODE != generate)')
'''))

# ─── Download standard negative datasets (DNS, AudioSet etc) ────────────
CELLS.append(code('''# === Download standard negative datasets (DNS challenge, AudioSet, MIT IRs) ===
# Pre-generated spectrogram features hosted on HuggingFace by kahrendt.
import os, subprocess
from huggingface_hub import snapshot_download

snapshot_download(
    'kahrendt/microwakeword',
    repo_type='dataset',
    local_dir='/content/negative_datasets',
    allow_patterns=['speech/*', 'dinner_party/*', 'no_speech/*',
                    'dinner_party_eval/*'],
)

# MIT room impulse responses for reverb augmentation
if not os.path.exists('mit_rirs') or not os.listdir('mit_rirs'):
    print('Downloading MIT room impulse responses...')
    subprocess.run(['mkdir', '-p', 'mit_rirs'], check=True)
    subprocess.run('cd mit_rirs && wget -q https://www.openslr.org/resources/28/rirs_noises.zip && unzip -q rirs_noises.zip',
                   shell=True, check=True)

# Background noise corpora (FMA + AudioSet 16 kHz subsets)
if not os.path.exists('fma_16k') or not os.listdir('fma_16k'):
    print('Downloading FMA background corpus (~500 MB)...')
    subprocess.run('mkdir -p fma_16k && cd fma_16k && wget -q https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/fma_16k.tar && tar -xf fma_16k.tar && rm fma_16k.tar',
                   shell=True, check=True)
if not os.path.exists('audioset_16k') or not os.listdir('audioset_16k'):
    print('Downloading AudioSet background corpus (~500 MB)...')
    subprocess.run('mkdir -p audioset_16k && cd audioset_16k && wget -q https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/audioset_16k.tar && tar -xf audioset_16k.tar && rm audioset_16k.tar',
                   shell=True, check=True)
print('All negative datasets ready')
'''))

# ─── Augmentation pipeline + feature generation ──────────────────────────
CELLS.append(code('''# === Augmentation + feature extraction ===
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration
import os, shutil, traceback
from mmap_ninja.ragged import RaggedMmap

clips = Clips(
    input_directory='generated_samples',
    file_pattern='*.wav',
    max_clip_duration_s=None,
    remove_silence=True,
    random_split_seed=42,
    split_count=0.1,
)
augmenter = Augmentation(
    augmentation_duration_s=3.2,
    augmentation_probabilities={
        'SevenBandParametricEQ': 0.15, 'TanhDistortion': 0.10,
        'PitchShift': 0.15, 'BandStopFilter': 0.10,
        'AddColorNoise': 0.20, 'AddBackgroundNoise': 0.85,
        'Gain': 1.00, 'GainTransition': 0.25, 'RIR': 0.60,
    },
    impulse_paths=['mit_rirs'],
    background_paths=['fma_16k', 'audioset_16k'],
    background_min_snr_db=-5, background_max_snr_db=20,
    min_jitter_s=0.10, max_jitter_s=0.50,
)

os.makedirs('generated_augmented_features', exist_ok=True)
SPLIT_CONFIG = {
    'training':   {'split_name': 'train',      'repetition': 3, 'slide_frames': 10},
    'validation': {'split_name': 'validation', 'repetition': 1, 'slide_frames': 10},
    'testing':    {'split_name': 'test',       'repetition': 1, 'slide_frames': 1 },
}

for split, cfg in SPLIT_CONFIG.items():
    out = f'generated_augmented_features/{split}'
    mmap = f'{out}/wakeword_mmap'
    if os.path.exists(mmap) and list(os.scandir(mmap)):
        print(f'{split}: cached, skipping')
        continue
    if os.path.exists(mmap):
        shutil.rmtree(mmap)
    os.makedirs(out, exist_ok=True)
    print(f'Generating {split} (rep={cfg["repetition"]}, slide={cfg["slide_frames"]})...')
    try:
        sg = SpectrogramGeneration(clips=clips, augmenter=augmenter,
                                    slide_frames=cfg['slide_frames'], step_ms=10)
        RaggedMmap.from_generator(
            out_dir=mmap, batch_size=200, verbose=True,
            sample_generator=sg.spectrogram_generator(
                split=cfg['split_name'], repeat=cfg['repetition']),
        )
    except Exception:
        traceback.print_exc()
        if os.path.exists(mmap): shutil.rmtree(mmap)
        raise
print('Positive features ready')

# Confusable features (only if confusable_negatives/ exists)
if os.path.exists('confusable_negatives') and os.listdir('confusable_negatives'):
    print('Generating confusable features...')
    confusable_clips = Clips(
        input_directory='confusable_negatives', file_pattern='*.wav',
        max_clip_duration_s=None, remove_silence=True,
        random_split_seed=42, split_count=0.1,
    )
    os.makedirs('confusable_features', exist_ok=True)
    for split, cfg in SPLIT_CONFIG.items():
        out = f'confusable_features/{split}'
        mmap = f'{out}/wakeword_mmap'
        if os.path.exists(mmap) and list(os.scandir(mmap)):
            continue
        if os.path.exists(mmap): shutil.rmtree(mmap)
        os.makedirs(out, exist_ok=True)
        sg = SpectrogramGeneration(clips=confusable_clips, augmenter=augmenter,
                                    slide_frames=cfg['slide_frames'], step_ms=10)
        RaggedMmap.from_generator(
            out_dir=mmap, batch_size=200, verbose=True,
            sample_generator=sg.spectrogram_generator(
                split=cfg['split_name'], repeat=cfg['repetition']),
        )
    print('Confusable features ready')

# Real recording features (only if real_recordings/ exists)
if os.path.exists('real_recordings') and os.listdir('real_recordings'):
    print('Generating real-recording features...')
    real_clips = Clips(
        input_directory='real_recordings', file_pattern='*.wav',
        max_clip_duration_s=None, remove_silence=True,
        random_split_seed=42, split_count=0.1,
    )
    os.makedirs('real_recording_features', exist_ok=True)
    for split, cfg in SPLIT_CONFIG.items():
        out = f'real_recording_features/{split}'
        mmap = f'{out}/wakeword_mmap'
        if os.path.exists(mmap) and list(os.scandir(mmap)):
            continue
        if os.path.exists(mmap): shutil.rmtree(mmap)
        os.makedirs(out, exist_ok=True)
        sg = SpectrogramGeneration(clips=real_clips, augmenter=augmenter,
                                    slide_frames=cfg['slide_frames'], step_ms=10)
        RaggedMmap.from_generator(
            out_dir=mmap, batch_size=200, verbose=True,
            sample_generator=sg.spectrogram_generator(
                split=cfg['split_name'], repeat=cfg['repetition']),
        )
    print('Real-recording features ready')
'''))

# ─── Training config YAML ────────────────────────────────────────────────
CELLS.append(code(f'''# === Training config YAML ===
import yaml, os
from pathlib import Path

SKIP_CONFUSABLES = not Path('confusable_features/training/wakeword_mmap').exists()
SKIP_REAL = not Path('real_recording_features/training/wakeword_mmap').exists()

config = {{
    'window_step_ms': 10,
    'train_dir': f'trained_models/{{OUTPUT_NAME}}',
    'features': [
        dict(features_dir='generated_augmented_features', sampling_weight=8.0,
             penalty_weight=2.0, truth=True, truncation_strategy='truncate_start',
             type='mmap'),
        dict(features_dir='negative_datasets/speech', sampling_weight=10.0,
             penalty_weight=2.5, truth=False, truncation_strategy='random', type='mmap'),
        dict(features_dir='negative_datasets/dinner_party', sampling_weight=15.0,
             penalty_weight=3.0, truth=False, truncation_strategy='random', type='mmap'),
        dict(features_dir='negative_datasets/no_speech', sampling_weight=5.0,
             penalty_weight=1.0, truth=False, truncation_strategy='random', type='mmap'),
        dict(features_dir='negative_datasets/dinner_party_eval', sampling_weight=0.0,
             penalty_weight=1.0, truth=False, truncation_strategy='split', type='mmap'),
    ],
    'training_steps': [25000, 20000],
    'positive_class_weight': [2, 2],
    'negative_class_weight': [40, 50],
    'learning_rates': [0.001, 0.0001],
    'batch_size': 256,
    'time_mask_max_size': [5, 5], 'time_mask_count': [1, 1],
    'freq_mask_max_size': [3, 3], 'freq_mask_count': [1, 1],
    'eval_step_interval': 500,
    'clip_duration_ms': 1500,
    'target_minimization': 0.4,
    'minimization_metric': 'ambient_false_positives_per_hour',
    'maximization_metric': 'average_viable_recall',
}}

if not SKIP_CONFUSABLES:
    config['features'].append(dict(features_dir='confusable_features', sampling_weight=8.0,
                                    penalty_weight=5.0, truth=False,
                                    truncation_strategy='random', type='mmap'))
if not SKIP_REAL:
    config['features'].append(dict(features_dir='real_recording_features', sampling_weight=8.0,
                                    penalty_weight=2.0, truth=True,
                                    truncation_strategy='truncate_start', type='mmap'))

os.makedirs(f'trained_models/{{OUTPUT_NAME}}', exist_ok=True)
with open('training_parameters.yaml', 'w') as f:
    yaml.dump(config, f)
print('training_parameters.yaml ready')
print(f'  Feature sets: {{len(config["features"])}}, total steps: {{sum(config["training_steps"])}}')
'''))

# ─── Train ──────────────────────────────────────────────────────────────
CELLS.append(code('''# === Train the model ===
# subprocess + PYTHONPATH so it can find microwakeword (sys.path doesn't propagate)
import os, sys, subprocess, shutil
shutil.rmtree(f'trained_models/{OUTPUT_NAME}', ignore_errors=True)

env = os.environ.copy()
env['PYTHONPATH'] = '/content/microWakeWord:' + env.get('PYTHONPATH', '')
env['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'

cmd = [
    sys.executable, '-m', 'microwakeword.model_train_eval',
    '--training_config', 'training_parameters.yaml',
    '--train', '1', '--restore_checkpoint', '0',
    '--test_tflite_streaming_quantized', '1',
    '--use_weights', 'best_weights',
    'mixednet',
    '--pointwise_filters', '64,64,64,64',
    '--repeat_in_block', '1, 1, 1, 1',
    '--mixconv_kernel_sizes', '[5], [7,11], [9,15], [23]',
    '--residual_connection', '0,0,0,0',
    '--first_conv_filters', '32',
    '--first_conv_kernel_size', '5',
    '--stride', '3',
]
print('Running:', ' '.join(cmd)); print()
proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end='')
proc.wait()
print(); print('Exit code:', proc.returncode)
assert proc.returncode == 0, 'training failed - see output above'
'''))

# ─── Export + push to Drive ──────────────────────────────────────────────
CELLS.append(code('''# === Export + push to Drive ===
import os, json, shutil, datetime

tflite_src = f'trained_models/{OUTPUT_NAME}/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite'
assert os.path.exists(tflite_src), f'No model at {tflite_src}'

OUT_TFLITE = f'{OUTPUT_NAME}.tflite'
OUT_JSON = f'{OUTPUT_NAME}.json'
shutil.copy2(tflite_src, OUT_TFLITE)
print(f'wrote {OUT_TFLITE} ({os.path.getsize(OUT_TFLITE)/1024:.1f} KB)')

manifest = {
    'type': 'micro',
    'wake_word': WAKE_WORD,
    'author': AUTHOR,
    'website': AUTHOR_WEBSITE,
    'model': OUT_TFLITE,
    'trained_languages': TRAINED_LANGUAGES,
    'version': 2,
    'micro': {
        'probability_cutoff': PROBABILITY_CUTOFF,
        'feature_step_size': 10,
        'sliding_window_size': SLIDING_WINDOW_SIZE,
        'tensor_arena_size': TENSOR_ARENA_SIZE,
        'minimum_esphome_version': '2024.7.0',
    }
}
with open(OUT_JSON, 'w') as f:
    json.dump(manifest, f, indent=2)
print(f'wrote {OUT_JSON}')
print(json.dumps(manifest, indent=2))

for fn in (OUT_TFLITE, OUT_JSON):
    shutil.copy2(fn, f'{DRIVE_DIR}/{fn}')
    print(f'pushed {fn} -> {DRIVE_DIR}')

ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
with open(f'{DRIVE_DIR}/_run_finished.txt', 'w') as f:
    f.write(f'Training run finished at {ts}\\n')
print()
print(f'DONE. Find your model at /content/drive/MyDrive/{DRIVE_FOLDER}/{OUTPUT_NAME}.tflite')
'''))

# ─── On-device deployment guide ──────────────────────────────────────────
CELLS.append(md("""## Deploying to ESPHome devices

Drop the `.tflite` + `.json` next to your ESPHome YAML (e.g. in `/config/esphome/wakewords/<output_name>/`).

In your device YAML, replace your existing wake word:

```yaml
micro_wake_word:
  models:
    - model: wakewords/<output_name>/<output_name>.json
  on_wake_word_detected:
    - voice_assistant.start:
```

Then `esphome run <device>.yaml` (USB or OTA).

## Manifest tuning (likely needed)

The defaults work for **Hey Harold** specifically. Your model's confidence
distribution will differ. Iterate:

| Symptom | Knob |
|---|---|
| Doesn't fire on the wake word | Lower `probability_cutoff` (try 0.7, 0.6) |
| Fires on too many things | Raise `probability_cutoff` (try 0.92, 0.95) |
| LED fires but no STT response | `sliding_window_size: 5` (faster fire); also check Echo speaker mute |
| `Failed to allocate tensors` log | Raise `tensor_arena_size` (try 50000, 80000) |

## Known gotchas
- Editable install (`pip install -e ./microWakeWord`) requires kernel restart — broken for Run All
- `train.py` upstream calls `.numpy()` on numpy arrays under newer TF (this notebook patches it)
- T4 GPU OOMs during validation — use A100 + High-RAM
- Manifest path mismatch: training writes to `trained_models/<output_name>/`, manifest must match
- `voice_assistant` has no audio lookback; if MWW fires AFTER user speaks, STT gets silence
"""))


def main():
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
            "accelerator": "GPU",
        },
        "cells": CELLS,
    }
    DEST.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Built: {DEST}")
    print(f"  cells: {len(CELLS)}, size: {DEST.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
