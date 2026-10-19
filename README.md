# Pocket TTS

Lightweight CPU-based Text-to-Speech extension for [text-generation-webui](https://github.com/oobabooga/text-generation-webui) using Kyutai's [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) model.

## Features

- **Lightweight**: ~100M parameter model (vs 1.6B for alternatives like Moshi)
- **CPU-only**: No GPU contention with your LLM
- **Fast**: ~7x faster than real time on an M2 Pro (real-time factor ~0.14)
- **Simple**: one small `script.py`, no external servers
- **7 languages**: English, French, German, Portuguese, Italian, Spanish, Dutch
- **14 built-in voices**: alba, marius, javert, jean, fantine, cosette, eponine, azelma, plus the upstream per-language defaults estelle, juergen, rafael, giovanni, lola, daan
- **Voice cloning**: custom WAV files, cloned once and cached as exported voice states

## Upgrading from pocket-tts 0.1 (this release)

This version targets **pocket-tts 3.x** (`pocket-tts>=3.3,<4`). Changes in the extension:

- New **Language** setting. Each language is a separate model (`TTSModel.load_model(language=...)`); switching reloads the model on the next message (a few seconds). Every built-in voice has an embedding for every language.
- Built-in voices now load from precomputed per-language embeddings, so they work **without** the gated voice-cloning weights.
- Custom voices are cloned once and the resulting voice state is exported with `export_model_state` to `voices/.cache/<voice>.<language>.safetensors`. Later uses load that file instead of re-encoding the WAV; the cache is rebuilt if the WAV is newer. Delete `voices/.cache/` to force a rebuild.
- A **Refresh voices** button picks up WAVs added to `voices/` without restarting.
- **numpy**: pocket-tts 3.x requires `numpy>=2`. Current text-generation-webui builds ship numpy 2.x (e.g. 2.2), so nothing extra is needed. The old advice to install `numpy<2` no longer applies; if you are on an old webui build pinned to numpy 1.x, update the webui first or stay on the previous release of this extension.

## Requirements

- text-generation-webui (a build with numpy 2.x)
- Python 3.10-3.14
- PyTorch 2.5+ (CPU build is fine)

## Installation

### Option 1: Git Clone (Recommended)

```bash
cd /path/to/text-generation-webui/extensions
git clone https://github.com/jschuller/pocket_tts.git
pip install -r pocket_tts/requirements.txt
```

Run `pip` with the webui's own environment (e.g. after `./cmd_macos.sh` / `cmd_linux.sh`).

### Option 2: Manual Download

1. Download and extract to `text-generation-webui/extensions/pocket_tts/`

2. **Install dependencies**:
   ```bash
   cd /path/to/text-generation-webui
   pip install -r extensions/pocket_tts/requirements.txt
   ```

3. **Enable the extension** by adding to `user_data/CMD_FLAGS.txt`:
   ```
   --extensions pocket_tts
   ```

4. **Restart** text-generation-webui

## Configuration

The extension adds a "Pocket TTS" accordion in the Chat tab with these options:

| Setting | Description | Default |
|---------|-------------|---------|
| **Activate TTS** | Enable/disable audio generation | On |
| **Play automatically** | Autoplay audio for new responses | On |
| **Show message text** | Display text below audio player | Off |
| **Language** | english, french, german, portuguese, italian, spanish, dutch | english |
| **Voice** | Built-in voice or a custom WAV from `voices/` | alba |

Upstream also publishes larger `*_24l` variants and dated English checkpoints; they are not exposed in the UI to keep it simple.

## Custom Voices (cloning)

> **Consent:** only clone voices you have explicit permission to use. Kyutai's terms prohibit voice impersonation or cloning without explicit and lawful consent, and presenting generated audio as genuine recordings of real people.

### Gated weights

Built-in voices use the ungated `kyutai/pocket-tts-without-voice-cloning` weights. Cloning from a WAV needs the gated `kyutai/pocket-tts` weights:

1. Accept the model terms at <https://huggingface.co/kyutai/pocket-tts>.
2. Log in on the machine running the webui: `hf auth login` (or `uvx hf auth login`).
3. Restart the webui. The console prints `voice cloning enabled` when the model loads.

Without them, pocket-tts falls back to the ungated weights automatically; built-in voices keep working and selecting a custom voice shows an error explaining the steps above.

### Adding a voice

1. Place a WAV file (10-30 seconds of clear speech; longer files are truncated to 30 s) in the `voices/` directory
2. Click **Refresh voices** (or restart); the voice appears under its file name
3. Select it and generate a response. The first use clones and caches the voice state; later uses are fast

You can also pre-export a state with the upstream CLI (`pocket-tts export-voice my_voice.wav my_voice.safetensors --language english`).

**Tips for good voice cloning:**
- Use high-quality audio (16kHz+ sample rate); its quality carries over to the output
- Clear speech without background noise
- 10-30 seconds of varied speech

## Usage

1. Start text-generation-webui with the extension enabled
2. Load any LLM model
3. Start a chat - responses will automatically generate audio
4. Use the Preview button to test voice settings

## Standalone Test

`test_tts.py` runs without the webui. It loads the model, synthesizes one sentence, writes `outputs/test_output.wav`, prints the duration and real-time factor, and checks an export/reload round trip of a voice state:

```bash
python test_tts.py                       # english, alba
python test_tts.py --language french --voice estelle
python test_tts.py --clone voices/me.wav # needs the gated weights
```

## OpenAI-compatible endpoint

This extension does not expose an HTTP API. If you need an OpenAI-style `/v1/audio/speech` endpoint for Pocket TTS, use a separate server instead, for example a local mlx-audio server with model id `mlx-community/pocket-tts`:

```bash
curl http://127.0.0.1:8800/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model": "mlx-community/pocket-tts", "input": "Hello there", "voice": "alba", "response_format": "wav"}' \
  -o speech.wav
```

Without `response_format` the server returns MP3. The first request loads the model (~20 s).

Upstream's `pocket-tts serve` also provides a local web UI and HTTP server (default port 8000).

## API

The extension hooks into these text-generation-webui extension points:

- `output_modifier()` - Generates audio for bot responses
- `state_modifier()` - Disables streaming (required for TTS)
- `history_modifier()` - Removes autoplay from old messages
- `ui()` - Adds configuration UI

## Troubleshooting

### "Voice cloning needs the gated Pocket TTS weights"
See [Gated weights](#gated-weights). Built-in voices work without them.

### numpy version conflicts
pocket-tts 3.x needs `numpy>=2`. Check with `python -c "import numpy; print(numpy.__version__)"` in the webui environment.

### First response is slow
The TTS model loads on first use and after a language change (~4-5 seconds, plus a one-time download). Subsequent responses are fast.

### No audio plays
- Check browser autoplay settings
- Verify the extension is enabled in Session tab
- Check console for errors

### Audio quality issues
- Try different voices (some work better for certain content)
- Match the Language setting to the language of the replies
- Shorter responses generally sound better

## Technical Details

- **Model**: Kyutai Pocket TTS 3.x (~100M parameters), runs on CPU
- **Sample Rate**: 24kHz
- **Output Format**: WAV
- **Voice State Caching**: in memory per (language, voice); cloned voices also on disk in `voices/.cache/`

## Files

```
pocket_tts/
├── script.py           # Main extension code
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── LICENSE             # MIT License
├── test_tts.py         # Standalone test script
├── test_sentences.txt  # Preview sentences
├── outputs/            # Generated audio files
└── voices/             # Custom voice WAV files (+ .cache/ exported states)
```

## Credits

- [Kyutai](https://github.com/kyutai-labs) for Pocket TTS
- [oobabooga](https://github.com/oobabooga) for text-generation-webui
- Based on the silero_tts extension pattern

## License

This extension's code is MIT licensed - see [LICENSE](LICENSE).

The Pocket TTS model weights (`kyutai/pocket-tts` and `kyutai/pocket-tts-without-voice-cloning`) are released by Kyutai under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/); attribute Kyutai when you distribute generated audio or derived work. The pocket-tts library is MIT licensed. Individual built-in voices come from their own sources (see upstream's voice licensing notes).
