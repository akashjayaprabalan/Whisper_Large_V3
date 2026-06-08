# Whisper large-v3 NVFP4 transcription + small English translation

This folder contains a local Python pipeline that:

1. Uses `Reza2kn/openai_whisper-large-v3-NVFP4` for speech-to-text transcription.
2. Translates the resulting transcript to English with a small text model.

The default translation model is `PontifexMaximus/opus-mt-iir-en-finetuned-fa-to-en`, a compact Marian model fine-tuned for Persian-to-English.

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
