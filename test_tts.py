"""Standalone test script for Pocket TTS."""
from pocket_tts import TTSModel
import scipy.io.wavfile
from pathlib import Path


def test_pocket_tts():
    print("Loading model...")
    model = TTSModel.load_model()

    print("Loading voice state for 'alba'...")
    voice = model.get_state_for_audio_prompt("alba")

    print("Generating audio...")
    text = "Hello, this is a test of Pocket TTS. The quick brown fox jumps over the lazy dog."
    audio = model.generate_audio(voice, text)

    output = Path(__file__).parent / "outputs" / "test_output.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(str(output), model.sample_rate, audio.numpy())

    print(f"Audio saved to: {output}")
    print(f"Sample rate: {model.sample_rate}")
    print(f"Audio length: {len(audio)} samples ({len(audio)/model.sample_rate:.2f}s)")
    print("SUCCESS!")


if __name__ == "__main__":
    test_pocket_tts()
