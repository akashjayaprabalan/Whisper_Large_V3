# Whisper large-v3 NVFP4 transcription + small English translation

This folder contains a local Python pipeline that:

1. Uses `Reza2kn/openai_whisper-large-v3-NVFP4` for speech-to-text transcription.
2. Translates the resulting transcript to English with a small text model.

The default translation model is `PontifexMaximus/opus-mt-iir-en-finetuned-fa-to-en`, a compact Marian model fine-tuned for Persian-to-English. If you want to compare against a larger mT5 translator, pass `--translation-model persiannlp/mt5-small-parsinlu-opus-translation_fa_en`.

The top-level scripts are thin command wrappers. The implementation lives in the
`whisper_large_v3/` package so ASR loading, translation chunking, device
selection, model downloads, and output writing can be tested independently.

## Setup

Use Python 3.11 on this machine:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Download the ASR and translation models into this project:

```bash
.venv/bin/python download_models.py
```

The ASR model is about 1.1 GB. The default translation model is about 304 MB.

## Run

```bash
.venv/bin/python transcribe_translate.py /path/to/audio.mp3 \
  --output-json outputs/result.json \
  --output-text outputs/english.txt
```

By default, Whisper is forced to Persian/Farsi with `--language fa`. For automatic language detection:

```bash
.venv/bin/python transcribe_translate.py /path/to/audio.mp3 --language auto
```

For the larger mT5 Persian-to-English translator:

```bash
.venv/bin/python transcribe_translate.py /path/to/audio.mp3 \
  --translation-model persiannlp/mt5-small-parsinlu-opus-translation_fa_en
```

To produce only the source-language transcript:

```bash
.venv/bin/python transcribe_translate.py /path/to/audio.mp3 --skip-translation
```

## Development checks

```bash
python3 -m unittest
python3 -m compileall .
python3 transcribe_translate.py --help
python3 download_models.py --help
```
