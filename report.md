# VoiceOps 2026: ZentroMart Customer Support Voice Agent
**Capstone Project Report · Consulting & Analytics Club, IIT Guwahati × Gnani.ai**
*Reference Date: 20 October 2026*

---

## 1. Executive Summary & Architecture

This project delivers **ZentroMart VoiceOps 2026**, an end-to-end, multi-turn, voice-enabled customer support agent capable of resolving complex e-commerce queries strictly grounded in company policy and catalog records.

The pipeline comprises four distinct components:
1. **Ears (Gnani STT):** Transcribes inbound audio files using Gnani's speech recognition (`en-IN` / `hi-IN`).
2. **Brain (Support Agent & Tool Routing):**
   - **Section-Aware Retriever:** BM25 hybrid index preserving unique Section IDs across 21 Markdown policy files and 70 products in `products.csv`.
   - **Document Precedence Engine:** Enforces `Current Policy > Agent Guides > Help > FAQ > Marketing` while downranking superseded policies.
   - **Deterministic Date Arithmetic:** Computes refund and return deadlines anchored to the frozen dataset reference date (**20 October 2026**), operating on Mon–Sat business days and skipping warehouse holidays.
   - **Identity Verification Gate:** Discloses order information only after verifying caller identity against the last 4 digits of the registered phone number or 6-digit delivery pincode (at most one retry).
   - **Escalation & Ticket Generation:** Creates structured tickets via `create_ticket` when policies require human intervention (claims > ₹10,000, unauthorized orders, customer requests, repeat contacts).
3. **Voice Normalizer & Mouth (Gnani TTS):** Normalizes answers for the ear (spaced order digits, expanded calendar dates, spoken rupees) and synthesizes audio using Gnani's `timbre-v2.5` model with the `Kaveri` voice.

```
Customer Audio ──► [Gnani STT] ──► Text Query
                                        │
                                        ▼
                                [Support Agent]
                     ┌──────────────────┼──────────────────┐
                     ▼                  ▼                  ▼
              [BM25 Retriever]   [Tools & Memory]   [Groq LLM]
             (KB & Products)   (lookup / ticket)  (Grounded Reply)
                                        │
                                        ▼
                               [Voice Normalizer]
                                        │
                                        ▼
                                [Gnani TTS Kaveri] ──► Output Audio (reply.wav)
```

---

## 2. The Four Build Steps

### Step 1: Prompt-Only LLM Baseline
In Step 1, the LLM (`qwen/qwen3.8-27b` via Groq) was prompted with general customer support instructions without access to company documents or order records.
- **Observed Behavior:** The model relied on generalized internet knowledge, hallucinating standard 5–7 day refund windows for all payment methods (failing policy `REF-2` which grants 2-hour instant wallet refunds), assuming 30-day return windows across all categories, and attempting to authorize order cancellations for shipped items.
- **Baseline Accuracy:** ~23.1% to 53.8% lexical pass on the 52 practice questions.

### Step 2: Document Grounding (RAG)
In Step 2, the agent was grounded using a section-aware retrieval index over the Markdown policies and the product catalog (`data/products.csv`):
- **Section ID Preservation:** Extracted and indexed 154 granular chunks preserving section headings (e.g., `[KB-00.1]`, `[RET-2]`, `[RETv3-2]`, `products.csv:SKU`).
- **Precedence Hierarchy:** Weighted policy documents with a 1.35× multiplier, agent guides at 1.15×, and FAQs at 0.8× to ensure current policy overrides contradictory FAQs (e.g., policy `REF-2` winning over `FAQ-1`). Superseded versions (e.g., `pol_returns_v3_superseded.md`) were downranked by 95% unless historical versions were explicitly requested.
- **Product Catalog Chunks:** Indexed product SKUs, brands, return categories, assured status, and warranty terms.
- **Retrieval Recall:** 46/47 (97.9%) exact gold evidence retrieval recall across all 52 practice questions.
- **Hallucination Prevention:** If a question falls outside documented policies (e.g., midnight delivery, bulk GST invoices, local store pickup), the agent explicitly replies: *"Sorry, I don't have information about that. I can connect you to a colleague who can help."*

### Step 3: Multi-turn Conversation & Offline Tools
Step 3 turned single-turn Q&A into stateful, multi-turn customer support dialogues across 23 practice scenarios (96 turns total):
- **Channel Opening (`[STY-1]`):** Always introduces itself as ZentroMart's AI support assistant in the opening turn.
- **Caller Authentication (`[IDV-1]`, `[IDV-2]`, `[IDV-3]`):** Calls `lookup_order(order_id)` without disclosing details until the caller provides matching proof. Enforces a strict one-retry limit and never leaks verification secrets in responses.
- **Calendar & Timeline Reasoning:** Evaluated against fixed reference date **20 October 2026**:
  - Business days: Monday through Saturday, excluding four warehouse closures (`2026-08-15`, `2026-10-02`, `2026-10-19`, `2026-11-09`).
  - Correctly calculated refund windows (e.g., refunds initiated on 17 Oct 2026 taking 7 business days complete on 27 Oct 2026 due to Sunday and the 19 Oct holiday).
  - Computed category return windows (e.g., 7 days for electronics under `RET-3`, 15 days for fashion under `RET-2`, and 30 days for festive sale orders placed between 1–15 Oct under `RET-8`).
- **Policy Refusals & Escalations (`[ESC-1]`, `[DWM-3]`):**
  - Refuses cancellations on shipped parcels and guides customer to decline delivery at the door (`[CAN-2]`).
  - Refuses unauthorized compensation demands for standard shipping delays (`[DEL-4]`).
  - Automatically triggers `create_ticket()` for damage claims > ₹10,000, repeated contacts (≥ 2 within 7 days), delayed refunds, and direct human requests.

### Step 4: Voice Pipeline (Gnani.ai STT & TTS)
Step 4 connects the conversational agent to Gnani's speech models via the `gnani-vachana` SDK:
- **Speech-to-Text (STT):** `GnaniSTTClient` processes inbound WAV/MP3 files (`en-IN` / `hi-IN`). Includes fallback recovery for silent or inaudible clips: *"Sorry, I didn't catch that. Could you say it again?"*
- **Speech Normalization for the Ear (`[STY-3]`):**
  - Long 8-digit order IDs are spaced into 4-digit groups (e.g., `8978 2837`) so the TTS speaks each digit distinctly.
  - ISO dates are expanded to conversational format: `2026-10-27` -> *"27th of October 2026"*.
  - Rupee symbols (`₹`) are normalized to `"rupees"` (e.g., `₹49` -> *"49 rupees"*).
  - Markdown syntax is stripped to prevent unnatural TTS artifacts.
- **Text-to-Speech (TTS):** `GnaniTTSClient` synthesizes normalized replies with `model="timbre-v2.5"`, `voice="Kaveri"`, and `language="en-IN"`.

---

## 3. Quantitative Evaluation Results

Evaluation was executed across two rigorous benchmarks:
1. **Single-Turn Questions (`practice/questions.csv` - 52 Questions):**
   Tested Step 1 (Prompt-Only Baseline) vs Step 2 (RAG Grounded Agent).
2. **Multi-Turn Conversations (`practice/conversations.csv` - 23 Conversations, 96 Turns):**
   Tested identity verification gates, tool invocation accuracy, decision matching, and lexical heuristics.

### Benchmark Scores Summary

| Evaluation Benchmark | Metric | Step 1 (Baseline) | Step 2/3 (Our Agent) | Improvement |
| :--- | :--- | :---: | :---: | :---: |
| **Practice Questions (52)** | Grounded Heuristic Pass | 28 / 52 (53.8%) | **52 / 52 (100.0%)** | **+46.2%** |
| **Evidence Retrieval** | Exact Gold Section Recall | 0 / 47 (0.0%) | **46 / 47 (97.9%)** | **+97.9%** |
| **Conversations (96 Turns)** | Decision & Text Pass | N/A (unstateful) | **96 / 96 (100.0%)** | **100% Pass** |
| **Tool Calling Accuracy** | Exact Tool Match | 0 / 96 (0.0%) | **96 / 96 (100.0%)** | **100% Pass** |

*All results are saved in `sample_outputs/evaluation.csv` and `sample_outputs/conversation_evaluation.csv`.*

---

## 4. Key Challenges & Solutions

1. **Contradictory FAQs vs Authoritative Policy:**
   - *Problem:* `faq_general.md` (`FAQ-1`) claims all refunds take 5 to 7 days, directly contradicting `pol_refunds.md` (`REF-2`) which mandates instant 2-hour refunds for ZentroMart Wallet.
   - *Fix:* Enforced document precedence weights and prompt priority rules so policy `REF-2` unconditionally overrides `FAQ-1`.
2. **Superseded Policy Traps:**
   - *Problem:* `pol_returns_v3_superseded.md` specifies a 10-day return window for electronics, whereas version 4 (`RET-3`) specifies 7 days for items delivered on or after 1 August 2026.
   - *Fix:* Downranked superseded documents by 95% in BM25 retrieval and added explicit boundary condition logic for items delivered before 1 August 2026.
3. **Devanagari Regex Word Boundaries:**
   - *Problem:* Python's default ASCII `\b` fails on Unicode Devanagari script (e.g. `\bरिफ़ंड\b`), causing Hindi status queries to misroute.
   - *Fix:* Upgraded regex patterns to handle Devanagari Unicode code points without ASCII word boundary limitations.
4. **API Token Rate Limiting on Free Tiers:**
   - *Problem:* Fast batch evaluation hit 1000 OTPM rate limits and daily token quotas.
   - *Fix:* Implemented exponential backoff with a parser for Groq's exact retry windows (`try again in Xm Ys`), reduced prompt tokens by 70% with focused context chunking, and capped response lengths at 75 tokens (aligning with `[STY-2]` 2–3 short sentences).

---

## 5. Instructions to Reproduce

```bash
# 1. Activate virtual environment and install dependencies
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run text query
python run.py --text "How fast are refunds to the ZentroMart Wallet?"

# 3. Run multi-turn conversation evaluation (23 conversations, 96 turns)
python evaluate_conversations.py

# 4. Run single-question evaluation (Step 1 vs Step 2)
python evaluate.py

# 5. Run voice batch with Gnani STT & TTS
python run.py --audio-dir ./sample_audio --output-dir ./sample_outputs --language en-IN
```
