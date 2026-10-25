"""Interactive Voice Chat with ZentroMart Support Agent.

Uses:
- Microphone recording (arecord)
- Gnani STT (Speech-to-Text)
- Grounded Support Agent (Brain)
- Gnani TTS (Kaveri Voice)
- Audio playback (paplay / aplay)
"""
import os, sys, subprocess, time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import SupportAgent
from src.voice import voice_turn

def main():
    print("=" * 60)
    print("🎙️  ZentroMart Live Voice Assistant (Gnani.ai)")
    print("=" * 60)
    
    agent = SupportAgent()
    temp_in = PROJECT_ROOT / "sample_outputs" / "temp_input.wav"
    temp_out = PROJECT_ROOT / "sample_outputs" / "temp_reply.wav"
    
    language = "en-IN"
    print("\nSelect language: 1. English (en-IN) [Default]  2. Hindi (hi-IN)")
    lang_choice = input("Enter choice (1/2, default 1): ").strip()
    if lang_choice == "2":
        language = "hi-IN"
        print("Language set to: Hindi (hi-IN)")
    else:
        print("Language set to: Indian English (en-IN)")

    print("\nInstructions:")
    print("- Press ENTER to record your voice question (5 seconds).")
    print("- Speak clearly into your microphone.")
    print("- Type 'q' and press ENTER to exit.\n")

    session_id = f"live-chat-{int(time.time())}"

    while True:
        try:
            cmd = input("\n[Press ENTER to Speak | 'q' to Quit]: ")
            if cmd.lower().strip() == 'q':
                print("Exiting Voice Assistant. Goodbye!")
                break

            rec_duration = 5
            print(f"🔴 Recording for {rec_duration} seconds... Bolna shuru kijiye!")
            
            # Record using arecord
            rec_cmd = ["arecord", "-d", str(rec_duration), "-f", "S16_LE", "-r", "16000", "-c", "1", str(temp_in)]
            subprocess.run(rec_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("⏹️ Recording stopped. Processing audio...")

            if not temp_in.exists() or temp_in.stat().st_size < 1000:
                print("⚠️ Audio capture failed or empty. Please check microphone.")
                continue

            # Process with STT -> Agent -> TTS
            res = voice_turn(temp_in, agent, session_id=session_id, language_code=language, output_path=temp_out)
            
            transcript = res.get('transcript', '').strip()
            reply = res.get('reply', '').strip()
            
            print(f"\n🗣️ Aapne bola (Transcribed): \"{transcript}\"")
            print(f"🤖 ZentroMart Agent: \"{reply}\"")
            
            if res.get('sources'):
                src_ids = [s['section_id'] for s in res['sources']]
                print(f"📚 Sources: {', '.join(src_ids)}")

            # Play audio reply
            print("🔊 Playing response...")
            play_cmd = ["paplay", str(temp_out)]
            try:
                subprocess.run(play_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                subprocess.run(["aplay", str(temp_out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting Voice Assistant. Goodbye!")
            break
        except Exception as e:
            print(f"Error during voice interaction: {e}")

if __name__ == "__main__":
    main()
