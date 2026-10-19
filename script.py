"""
Pocket TTS Extension for text-generation-webui

A lightweight CPU-based TTS using Kyutai's Pocket TTS (pocket-tts 3.x, ~100M params).
Generates audio for bot responses, in several languages, with optional voice cloning.
"""
import html
import re
import threading
import time
from pathlib import Path
from typing import Any

import gradio as gr
import scipy.io.wavfile

from modules import chat, shared, ui_chat
from modules.utils import gradio

# Extension parameters (persisted via shared.settings)
params = {
    'activate': True,
    'voice': 'alba',
    'language': 'english',
    'autoplay': True,
    'show_text': False,
}

# Pre-made voices. Every voice has an embedding for every language; the first eight are the
# original English voices, the rest are the upstream default voices for the other languages.
BUILTIN_VOICES = [
    'alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma',
    'estelle', 'juergen', 'rafael', 'giovanni', 'lola', 'daan',
]
LANGUAGES = ['english', 'french', 'german', 'portuguese', 'italian', 'spanish', 'dutch']

EXTENSION_DIR = Path(__file__).resolve().parent
VOICES_DIR = EXTENSION_DIR / 'voices'           # custom WAVs to clone
VOICE_CACHE_DIR = VOICES_DIR / '.cache'          # exported voice states (.safetensors)
OUTPUT_DIR = Path('extensions/pocket_tts/outputs')  # relative to webui root: served as file/...

CLONING_HELP = (
    "Voice cloning needs the gated Pocket TTS weights: accept the terms at "
    "https://huggingface.co/kyutai/pocket-tts, run `hf auth login`, then restart the webui. "
    "Built-in voices work without it."
)

# Model and voice state cache (one model at a time, keyed by language)
_model: Any = None
_model_language: str | None = None
_voice_states: dict[tuple[str, str], Any] = {}
_lock = threading.Lock()  # generate_audio is not thread-safe


def load_model(language: str | None = None) -> Any:
    """Load the Pocket TTS model for a language (lazy, cached; reloads on language change)."""
    global _model, _model_language
    language = language or params['language']
    if _model is None or _model_language != language:
        from pocket_tts import TTSModel
        print(f"Loading Pocket TTS model ({language})...")
        _model = TTSModel.load_model(language=language)  # always on CPU
        _model_language = language
        _voice_states.clear()
        cloning = 'enabled' if _model.has_voice_cloning else 'unavailable (ungated weights)'
        print(f"Pocket TTS model loaded; voice cloning {cloning}")
    return _model


def list_voices() -> list[str]:
    """Built-in voices followed by any custom WAVs in voices/."""
    custom = sorted(p.stem for p in VOICES_DIR.glob('*.wav') if p.stem not in BUILTIN_VOICES)
    return BUILTIN_VOICES + custom


def _custom_voice_state(model: Any, wav: Path, language: str) -> Any:
    """Clone a voice from a WAV once, export the state, and reuse the export afterwards."""
    from pocket_tts import export_model_state
    cached = VOICE_CACHE_DIR / f'{wav.stem}.{language}.safetensors'
    if cached.exists() and cached.stat().st_mtime >= wav.stat().st_mtime:
        return model.get_state_for_audio_prompt(cached)
    if not model.has_voice_cloning:
        raise RuntimeError(CLONING_HELP)
    print(f"Cloning voice from {wav.name} (first use, then cached)...")
    state = model.get_state_for_audio_prompt(wav, truncate=True)
    VOICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    export_model_state(state, cached)
    return state


def get_voice_state(voice: str) -> Any:
    """Get or cache the voice state for the current language."""
    model = load_model()
    key = (_model_language, voice)
    if key not in _voice_states:
        wav = VOICES_DIR / f'{voice}.wav'
        if voice not in BUILTIN_VOICES and wav.exists():
            _voice_states[key] = _custom_voice_state(model, wav, _model_language)
        else:
            if voice not in BUILTIN_VOICES:
                print(f"Pocket TTS: voice '{voice}' not found, using 'alba'")
                voice = 'alba'
            _voice_states[key] = model.get_state_for_audio_prompt(voice)
    return _voice_states[key]


def synthesize(text: str, output_file: Path) -> None:
    """Generate speech for text with the selected voice and write it as a WAV file."""
    with _lock:
        model = load_model()
        voice_state = get_voice_state(params['voice'])
        audio = model.generate_audio(voice_state, text)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(str(output_file), model.sample_rate, audio.numpy())


def preprocess_text(text: str) -> str:
    """Clean text for TTS (remove markdown, HTML entities, etc.)."""
    # Unescape HTML entities
    text = html.unescape(text)

    # Remove markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic
    text = re.sub(r'__([^_]+)__', r'\1', text)      # Bold alt
    text = re.sub(r'_([^_]+)_', r'\1', text)        # Italic alt
    text = re.sub(r'~~([^~]+)~~', r'\1', text)      # Strikethrough
    text = re.sub(r'`([^`]+)`', r'\1', text)        # Code

    # Remove action text in asterisks (common in RP)
    text = re.sub(r'\*[^*]+\*', '', text)

    # Clean up whitespace
    text = ' '.join(text.split())

    return text.strip()


def state_modifier(state):
    """Disable streaming when TTS is active (TTS blocks on I/O)."""
    if not params['activate']:
        return state
    state['stream'] = False
    return state


def input_modifier(string, state):
    """Update processing message when TTS is active."""
    if not params['activate']:
        return string
    shared.processing_message = "*Is recording a voice message...*"
    return string


def history_modifier(history):
    """Remove autoplay from previous messages to prevent cascading playback."""
    if len(history['internal']) > 0:
        history['visible'][-1] = [
            history['visible'][-1][0],
            history['visible'][-1][1].replace('controls autoplay>', 'controls>')
        ]
    return history


def output_modifier(string, state):
    """Generate TTS audio for bot responses."""
    if not params['activate']:
        return string

    original_string = string
    text = preprocess_text(string)

    if not text:
        return '*Empty response*' if not original_string.strip() else original_string

    try:
        character = state.get('character_menu', 'unknown')
        output_file = OUTPUT_DIR / f'{character}_{int(time.time())}.wav'
        synthesize(text, output_file)

        # Build HTML audio element
        autoplay = 'autoplay' if params['autoplay'] else ''
        result = f'<audio src="file/{output_file.as_posix()}" controls {autoplay}></audio>'

        if params['show_text']:
            result += f'\n\n{original_string}'

        shared.processing_message = "*Is typing...*"
        return result

    except Exception as e:
        print(f"Pocket TTS error: {e}")
        shared.processing_message = "*Is typing...*"
        return original_string


def setup():
    """Initialize the extension (called once at startup)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if params['language'] not in LANGUAGES:
        params['language'] = 'english'


def remove_tts_from_history(history):
    """Replace audio elements with original text in history."""
    for i, entry in enumerate(history['internal']):
        history['visible'][i] = [history['visible'][i][0], entry[1]]
    return history


def toggle_text_in_history(history):
    """Toggle text visibility under audio elements."""
    for i, entry in enumerate(history['visible']):
        visible_reply = entry[1]
        if visible_reply.startswith('<audio'):
            if params['show_text']:
                reply = history['internal'][i][1]
                history['visible'][i] = [
                    history['visible'][i][0],
                    f"{visible_reply.split('</audio>')[0]}</audio>\n\n{reply}"
                ]
            else:
                history['visible'][i] = [
                    history['visible'][i][0],
                    f"{visible_reply.split('</audio>')[0]}</audio>"
                ]
    return history


def voice_preview(text: str) -> str:
    """Generate preview audio for testing voice settings."""
    if not text:
        text = "Hello! This is a preview of the selected voice."

    try:
        output_file = OUTPUT_DIR / 'voice_preview.wav'
        synthesize(text, output_file)
        return f'<audio src="file/{output_file.as_posix()}?{int(time.time())}" controls autoplay></audio>'
    except Exception as e:
        return f'Error: {e}'


def ui():
    """Create Gradio UI components."""
    with gr.Accordion("Pocket TTS"):
        with gr.Row():
            activate = gr.Checkbox(value=params['activate'], label='Activate TTS')
            autoplay = gr.Checkbox(value=params['autoplay'], label='Play TTS automatically')

        show_text = gr.Checkbox(value=params['show_text'], label='Show message text under audio player')

        with gr.Row():
            language = gr.Dropdown(value=params['language'], choices=LANGUAGES, label='Language')
            voice = gr.Dropdown(value=params['voice'], choices=list_voices(), label='Voice')
            refresh_voices = gr.Button('Refresh voices')

        with gr.Row():
            preview_text = gr.Textbox(
                show_label=False,
                placeholder="Preview text (leave empty for default)",
                elem_id="pocket_tts_preview_text"
            )
            preview_play = gr.Button("Preview")
            preview_audio = gr.HTML(visible=False)

        with gr.Row():
            convert = gr.Button('Permanently replace audios with message texts')
            convert_cancel = gr.Button('Cancel', visible=False)
            convert_confirm = gr.Button('Confirm (cannot be undone)', variant="stop", visible=False)

    # Convert history with confirmation
    convert_arr = [convert_confirm, convert, convert_cancel]
    convert.click(
        lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)],
        None, convert_arr
    )
    convert_confirm.click(
        lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)],
        None, convert_arr
    ).then(
        remove_tts_from_history, gradio('history'), gradio('history')
    ).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None
    ).then(
        chat.redraw_html, gradio(ui_chat.reload_arr), gradio('display')
    )
    convert_cancel.click(
        lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)],
        None, convert_arr
    )

    # Toggle message text in history
    show_text.change(
        lambda x: params.update({"show_text": x}), show_text, None
    ).then(
        toggle_text_in_history, gradio('history'), gradio('history')
    ).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None
    ).then(
        chat.redraw_html, gradio(ui_chat.reload_arr), gradio('display')
    )

    # Parameter update handlers (voice states are cached per (language, voice))
    activate.change(lambda x: params.update({"activate": x}), activate, None)
    autoplay.change(lambda x: params.update({"autoplay": x}), autoplay, None)
    voice.change(lambda x: params.update({"voice": x}), voice, None)
    language.change(lambda x: params.update({"language": x}), language, None)  # model reloads on next use
    refresh_voices.click(lambda: gr.update(choices=list_voices()), None, voice)

    # Preview handlers
    preview_text.submit(voice_preview, preview_text, preview_audio)
    preview_play.click(voice_preview, preview_text, preview_audio)
