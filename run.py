import argparse, json, os
from pathlib import Path
from src.agent import SupportAgent

def main():
    parser=argparse.ArgumentParser(description='ZentroMart VoiceOps capstone runner')
    parser.add_argument('--audio-dir',help='Folder of WAV audio files for Gnani STT/TTS batch processing')
    parser.add_argument('--output-dir',default='sample_outputs',help='Where output JSON/WAV files are saved')
    parser.add_argument('--text',help='Run a single text turn instead of audio batch')
    parser.add_argument('--session',default='cli')
    parser.add_argument('--language',default='en-IN')
    args=parser.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    agent=SupportAgent()
    if args.text:
        result=agent.respond(args.text,args.session); print(json.dumps(result,ensure_ascii=False,indent=2)); return
    if not args.audio_dir: parser.error('Provide --audio-dir or --text')
    from src.voice import voice_turn
    files=sorted(p for p in Path(args.audio_dir).glob('*') if p.suffix.lower() in {'.wav','.mp3','.amr'})
    if not files: parser.error('No supported audio files found in --audio-dir')
    for audio in files:
        wav=out/f'{audio.stem}_reply.wav'
        try:
            result=voice_turn(audio,agent,session_id=audio.stem,language_code=args.language,output_path=wav)
        except Exception as exc:
            result={'transcript':'','reply':f'ERROR: {exc}','sources':[],'audio_path':None}
        result['input_audio']=str(audio)
        (out/f'{audio.stem}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f"{audio.name}: {result.get('transcript','')} -> {result.get('reply','')}")
if __name__=='__main__': main()
