"""
Pocket TTS Extension for text-generation-webui

A lightweight CPU-based TTS using Kyutai's Pocket TTS (100M params).
Generates audio for bot responses with voice cloning support.
"""
import html
import re
import time
from pathlib import Path

import gradio as gr
import scipy.io.wavfile

from modules import chat, shared, ui_chat
from modules.utils import gradio

# Extension parameters (persisted via shared.settings)
params = {
    'activate': True,
    'voice': 'alba',
    'autoplay': True,
    'show_text': False,
}

# Built-in voice options
BUILTIN_VOICES = ['alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma']

# Model and voice state cache
_model = None
_voice_states = {}


def load_model():
    """Load the Pocket TTS model (lazy loading, cached)."""
    global _model
    if _model is None:
        from pocket_tts import TTSModel
        print("Loading Pocket TTS model...")
        _model = TTSModel.load_model()
        print("Pocket TTS model loaded successfully")
    return _model


def get_voice_state(voice: str):
    """Get or cache voice state for a given voice."""
    if voice not in _voice_states:
        model = load_model()
        print(f"Loading voice state for '{voice}'...")
        _voice_states[voice] = model.get_state_for_audio_prompt(voice)
    return _voice_states[voice]


def clear_voice_cache():
    """Clear the voice state cache (call when voice changes)."""
    global _voice_states
    _voice_states = {}


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

    # Preprocess text
    text = preprocess_text(string)

    if not text:
        return '*Empty response*' if not original_string.strip() else original_string

    try:
        # Generate audio
        model = load_model()
        voice_state = get_voice_state(params['voice'])
        audio = model.generate_audio(voice_state, text)

        # Save WAV file
        character = state.get('character_menu', 'unknown')
        output_file = Path(f'extensions/pocket_tts/outputs/{character}_{int(time.time())}.wav')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        scipy.io.wavfile.write(str(output_file), model.sample_rate, audio.numpy())

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
    # Create outputs directory if it doesn't exist
    Path('extensions/pocket_tts/outputs').mkdir(parents=True, exist_ok=True)


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


def voice_preview(text):
    """Generate preview audio for testing voice settings."""
    if not text:
        text = "Hello! This is a preview of the selected voice."

    try:
        model = load_model()
        voice_state = get_voice_state(params['voice'])
        audio = model.generate_audio(voice_state, text)

        output_file = Path('extensions/pocket_tts/outputs/voice_preview.wav')
        scipy.io.wavfile.write(str(output_file), model.sample_rate, audio.numpy())

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

        voice = gr.Dropdown(
            value=params['voice'],
            choices=BUILTIN_VOICES,
            label='Voice'
        )

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

    # Parameter update handlers
    activate.change(lambda x: params.update({"activate": x}), activate, None)
    autoplay.change(lambda x: params.update({"autoplay": x}), autoplay, None)

    def on_voice_change(x):
        params.update({"voice": x})
        clear_voice_cache()  # Clear cache so new voice loads

    voice.change(on_voice_change, voice, None)

    # Preview handlers
    preview_text.submit(voice_preview, preview_text, preview_audio)
    preview_play.click(voice_preview, preview_text, preview_audio)
