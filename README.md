# Pocket TTS

Lightweight CPU-based Text-to-Speech extension for [text-generation-webui](https://github.com/oobabooga/text-generation-webui) using Kyutai's Pocket TTS model.

## Features

- **Lightweight**: 100M parameter model (vs 1.6B for alternatives like Moshi)
- **CPU-only**: No GPU contention with your LLM
- **Fast**: 3-4x faster than realtime speech generation
- **Simple**: ~220 lines of code, no external servers
- **8 Built-in Voices**: alba, marius, javert, jean, fantine, cosette, eponine, azelma
- **Voice Cloning**: Support for custom voice WAV files

## Requirements

- text-generation-webui
- Python 3.10+
- numpy < 2.0 (required for text-generation-webui compatibility)

## Installation

### Option 1: Git Clone (Recommended)

```bash
cd /path/to/text-generation-webui/extensions
git clone https://github.com/jschuller/pocket_tts.git
pip install -r pocket_tts/requirements.txt
```

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
| **Voice** | Select from 8 built-in voices | alba |

## Custom Voices

To use your own voice for cloning:

1. Place a WAV file (10-30 seconds of clear speech) in the `voices/` directory
2. The voice will appear in the dropdown menu
3. Select it and generate a response

**Tips for good voice cloning:**
- Use high-quality audio (16kHz+ sample rate)
- Clear speech without background noise
- 10-30 seconds of varied speech

## Usage

1. Start text-generation-webui with the extension enabled
2. Load any LLM model
3. Start a chat - responses will automatically generate audio
4. Use the Preview button to test voice settings

## API

The extension hooks into these text-generation-webui extension points:

- `output_modifier()` - Generates audio for bot responses
- `state_modifier()` - Disables streaming (required for TTS)
- `history_modifier()` - Removes autoplay from old messages
- `ui()` - Adds configuration UI

## Troubleshooting

### "numpy.dtype size changed" error
Install numpy < 2.0:
```bash
pip install "numpy<2.0"
```

### First response is slow
The TTS model loads on first use (~2-3 seconds). Subsequent responses are fast.

### No audio plays
- Check browser autoplay settings
- Verify the extension is enabled in Session tab
- Check console for errors

### Audio quality issues
- Try different voices (some work better for certain content)
- Shorter responses generally sound better
- The model works best with English text

## Technical Details

- **Model**: Kyutai Pocket TTS (100M parameters)
- **Sample Rate**: 24kHz
- **Output Format**: WAV
- **Voice State Caching**: Voice embeddings are cached for performance

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
└── voices/             # Custom voice WAV files
```

## Credits

- [Kyutai](https://github.com/kyutai-labs) for Pocket TTS
- [oobabooga](https://github.com/oobabooga) for text-generation-webui
- Based on the silero_tts extension pattern

## License

MIT License - see [LICENSE](LICENSE) file
