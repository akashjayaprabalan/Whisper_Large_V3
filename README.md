# Whisper large-v3 NVFP4 transcription + small English translation

This folder contains a local Python pipeline that:

1. Uses `Reza2kn/openai_whisper-large-v3-NVFP4` for speech-to-text transcription.
2. Translates the resulting transcript to English with a small text model.

The default translation model is `PontifexMaximus/opus-mt-iir-en-finetuned-fa-to-en`, a compact Marian model fine-tuned for Persian-to-English.
