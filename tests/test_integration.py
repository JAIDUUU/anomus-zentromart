"""Live integration test suite for Groq LLM and Gnani.ai STT/TTS.

Only runs when valid API keys are configured in the environment.
Gracefully skips tests when keys are unavailable or not configured,
documenting the exact procedure to run after obtaining keys.
"""
import unittest, os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import SupportAgent
from src.voice import voice_turn

class TestLiveIntegration(unittest.TestCase):
    def test_groq_live_response(self):
        """Integration test for Groq LLM grounding."""
        key = os.getenv('GROQ_API_KEY', '').strip()
        if not key or key == 'replace_with_your_groq_api_key':
            self.skipTest("GROQ_API_KEY is not configured; skipping live Groq integration test.")

        try:
            agent = SupportAgent()
            # Fast test with 1-turn call
            res = agent.respond("How fast are refunds to the ZentroMart Wallet?", session_id="test-live-groq")
            self.assertIn('answer', res)
            self.assertTrue(len(res['answer']) > 0)
            self.assertIn('sources', res)
            self.assertTrue(any('REF-2' in s['section_id'] for s in res['sources']))
        except Exception as e:
            if 'Rate limit' in str(e) or '429' in str(e) or 'tokens per day' in str(e):
                self.skipTest(f"Groq API rate limit reached: {e}")
            else:
                self.fail(f"Groq live API call failed: {e}")

    def test_gnani_live_stt_tts(self):
        """Integration test for Gnani.ai STT and TTS."""
        key = os.getenv('GNANI_API_KEY', '').strip()
        if not key or key == 'replace_with_your_gnani_api_key':
            self.skipTest("GNANI_API_KEY is not currently available; skipping live Gnani voice API test.")

        # Test audio file if exists
        sample_audio = PROJECT_ROOT / 'sample_audio' / 'test_query.wav'
        if not sample_audio.exists():
            self.skipTest("Sample customer audio file not found for live Gnani voice test.")

        agent = SupportAgent()
        out_wav = PROJECT_ROOT / 'sample_outputs' / 'test_live_reply.wav'
        result = voice_turn(sample_audio, agent, output_path=out_wav)
        self.assertIn('transcript', result)
        self.assertIn('reply', result)
        self.assertTrue(out_wav.exists())

if __name__ == '__main__':
    unittest.main()
