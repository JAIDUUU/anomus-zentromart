"""Gnani-only STT/TTS adapter with text-normalization for ear.

Features:
- Enforces Gnani API only (no third-party speech tools).
- Normalizes replies for the ear:
  - Formats order numbers into space-separated digit groups (e.g., '8 9 7 8 2 8 3 7').
  - Expands ISO dates (e.g. '2026-10-27' -> '27th of October 2026').
  - Converts rupee symbols ('₹49' -> '49 rupees').
  - Strips markdown formatting for clean vocalization.
- Graceful fallback for silent or unclear audio input.
- Uses Gnani timbre-v2.5 with Kaveri voice.
"""
import os, re
from pathlib import Path
from gnani.stt import GnaniSTTClient
from gnani.tts import GnaniTTSClient

def normalize_for_speech(text):
    if not text:
        return ""
    # 1. Format order IDs: break 8-digit numbers into 4-digit groups with spaces
    text = re.sub(r'(?<!\d)(\d{4})(\d{4})(?!\d)', r'\1 \2', text)
    # 2. Format ISO dates into spoken English dates
    def _repl_date(m):
        y, month, d = m.group(1), int(m.group(2)), int(m.group(3))
        months = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
        suffix = 'th' if 11 <= d <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(d % 10, 'th')
        return f"{d}{suffix} of {months[month-1]} {y}"
    text = re.sub(r'(\d{4})-(\d{2})-(\d{2})', _repl_date, text)
    # 3. Replace currency symbol with words
    text = re.sub(r'₹\s*(\d[\d,]*)', r'\1 rupees', text)
    # 4. Remove Markdown formatting
    text = re.sub(r'[*_#`\[\]]', '', text)
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def voice_turn(audio_path, agent, session_id='voice-default', language_code='en-IN', output_path='reply.wav'):
    if not os.getenv('GNANI_API_KEY'):
        raise RuntimeError('GNANI_API_KEY is missing. Add GNANI_API_KEY to your .env file.')

    stt = GnaniSTTClient()
    tts = GnaniTTSClient()

    try:
        transcript_resp = stt.transcribe(str(audio_path), language_code=language_code)
        heard = transcript_resp.get('transcript', '').strip()
    except Exception as exc:
        heard = ""

    if not heard:
        reply = "Sorry, I didn't catch that. Could you say it again?"
        sources = []
        calls = []
    else:
        result = agent.respond(heard, session_id=session_id)
        reply = result.get('answer', '')
        sources = result.get('sources', [])
        calls = result.get('tool_calls', [])

    spoken_text = normalize_for_speech(reply)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    tts.synthesize(
        spoken_text,
        voice='Kaveri',
        model='timbre-v2.5',
        language=language_code,
        output_file=str(out_p)
    )

    return {
        'transcript': heard,
        'reply': reply,
        'spoken_text': spoken_text,
        'sources': sources,
        'tool_calls': calls,
        'audio_path': str(out_p)
    }
