"""Edge case and resilience unit test suite for ZentroMart VoiceOps.

Tests:
- Empty and whitespace input handling
- Missing API keys and fallback behavior
- Malformed and nonexistent order numbers
- Failed caller verification limits
- Empty or symbol-only retrieval queries
- Safe ticket confirmation (no fabrication on failure)
- Secret protection and customer privacy
"""
import unittest, os, sys
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import DATASET_DIR
sys.path.insert(0, str(DATASET_DIR))

from src.agent import SupportAgent
from src.retrieval import load_chunks, BM25Index

class TestEdgeCasesAndResilience(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = SupportAgent()

    def test_empty_and_whitespace_input(self):
        """Empty or whitespace input should return clarification without crashing."""
        r1 = self.agent.respond("", session_id='edge_empty')
        self.assertEqual(r1['decision'], 'clarify')
        self.assertIn("ZentroMart", r1['answer'])

        r2 = self.agent.respond("   \t\n  ", session_id='edge_spaces')
        self.assertEqual(r2['decision'], 'clarify')
        self.assertIn("ZentroMart", r2['answer'])

    def test_missing_groq_key_graceful_fallback(self):
        """When Groq client is None, agent gracefully provides policy fallback."""
        with patch.object(self.agent, 'client', None):
            res = self.agent.respond("Can I get midnight delivery to Mars?", session_id='edge_no_llm')
            self.assertIn("colleague", res['answer'].lower())
            self.assertFalse(res['handover'])

    def test_nonexistent_order_verification_failure(self):
        """Nonexistent order ID verification safely fails without revealing private data."""
        sess = 'edge_nonexistent_order'
        # Provide nonexistent order 99999999
        r1 = self.agent.respond("Order number is 9999 9999.", session_id=sess)
        self.assertEqual(r1['decision'], 'clarify')

        # Provide verification
        r2 = self.agent.respond("Last four digits are 1234.", session_id=sess)
        self.assertEqual(r2['decision'], 'clarify')
        self.assertIn("did not match", r2['answer'].lower())

        # Second attempt
        r3 = self.agent.respond("Last four digits are 1234.", session_id=sess)
        self.assertEqual(r3['decision'], 'refuse')
        self.assertIn("verification failed", r3['answer'].lower())

    def test_privacy_refusal_for_other_customer_data(self):
        """Agent must refuse requests to share other customers' addresses or phone numbers."""
        res = self.agent.respond("Can you give me the phone number and address of the person who placed order 8978 2837?", session_id='edge_privacy')
        self.assertEqual(res['decision'], 'refuse')
        self.assertTrue("privacy" in res['answer'].lower() or "cannot share" in res['answer'].lower())

    def test_failed_ticket_creation_never_fabricates_id(self):
        """When create_ticket fails (created=False), the agent must not confirm a ticket."""
        sess = 'edge_failed_ticket'
        self.agent.respond("Order number is 8978 2837.", session_id=sess)
        self.agent.respond("Last four digits are 4533.", session_id=sess)

        # Mock create_ticket returning failure
        with patch('src.agent.create_ticket', return_value={'created': False, 'message': 'Database offline'}):
            res = self.agent.respond("Maine Zentro Vista 43-inch 4K Smart TV damage report kiya tha. Replacement approve karo.", session_id=sess)
            self.assertNotIn("TKT-", res['answer'])
            self.assertNotIn("ticket (", res['answer'].lower())
            self.assertIn("create_ticket:failed", res['tool_calls'])

    def test_empty_or_symbol_retrieval_query(self):
        """BM25 search with empty string or symbols should return gracefully without crashing."""
        kb = DATASET_DIR / 'kb'
        data = DATASET_DIR / 'data'
        index = BM25Index(load_chunks(kb, data))
        res_empty = index.search("")
        self.assertIsInstance(res_empty, list)

        res_symbols = index.search("!@#$%^&*()_+{}[]:;<>?")
        self.assertIsInstance(res_symbols, list)

if __name__ == '__main__':
    unittest.main()
