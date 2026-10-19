"""Standalone smoke test for Pocket TTS 3.x (runs without text-generation-webui).

Usage: python test_tts.py [--language english] [--voice alba] [--clone path/to/voice.wav]
"""
import argparse
import tempfile
import time
from pathlib import Path

import scipy.io.wavfile
from pocket_tts import TTSModel, export_model_state

TEXT = "Hello, this is a test of Pocket TTS. The quick brown fox jumps over the lazy dog."


def timed_generate(model: TTSModel, state, text: str) -> tuple:
    start = time.perf_counter()
    audio = model.generate_audio(state, text)
    return audio, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', default='english')
    parser.add_argument('--voice', default='alba')
    parser.add_argument('--clone', type=Path, help='optional WAV to clone (needs gated weights)')
    args = parser.parse_args()

    start = time.perf_counter()
    model = TTSModel.load_model(language=args.language)
    print(f"Model ({args.language}) loaded in {time.perf_counter() - start:.2f}s on {model.device}; "
          f"sample rate {model.sample_rate} Hz; voice cloning: {model.has_voice_cloning}")

    voice = model.get_state_for_audio_prompt(args.voice)
    model.generate_audio(voice, "Warm up.")  # first call pays one-off setup costs
    audio, elapsed = timed_generate(model, voice, TEXT)

    output = Path(__file__).parent / "outputs" / "test_output.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(str(output), model.sample_rate, audio.numpy())
    duration = audio.shape[-1] / model.sample_rate
    print(f"Voice '{args.voice}': {duration:.2f}s of audio in {elapsed:.2f}s "
          f"(real-time factor {elapsed / duration:.3f}, {duration / elapsed:.1f}x faster than real time)")
    print(f"Saved to {output}")

    # Exported voice state round trip (how the extension caches cloned voices)
    with tempfile.TemporaryDirectory() as tmp:
        cached = Path(tmp) / f"{args.voice}.safetensors"
        export_model_state(voice, cached)
        reloaded = model.get_state_for_audio_prompt(cached)
        audio2, _ = timed_generate(model, reloaded, "This voice was reloaded from an exported state.")
        assert audio2.numel() > 0
        print(f"Exported state round trip OK ({cached.stat().st_size / 1e6:.1f} MB)")

    if args.clone:
        if not model.has_voice_cloning:
            print("Skipping --clone: gated weights unavailable (accept terms on HF + `hf auth login`).")
        else:
            cloned = model.get_state_for_audio_prompt(args.clone, truncate=True)
            audio3, _ = timed_generate(model, cloned, TEXT)
            print(f"Cloned voice OK: {audio3.shape[-1] / model.sample_rate:.2f}s")

    print("SUCCESS!")


if __name__ == "__main__":
    main()
