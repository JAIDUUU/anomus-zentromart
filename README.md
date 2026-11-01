# ZentroMart VoiceOps 2026

Customer-support agent for the From Tokens to Transformers capstone. The same agent supports grounded text answers, source-tracked retrieval, verified mock order lookups, ticket handovers, and Gnani.ai speech input/output.

## Requirements

- Python 3.10+
- Groq API key for LLM calls
- Gnani.ai API key for voice mode. Voice must use Gnani STT and TTS; no alternative speech provider is used.
- The supplied `Capstone_Project_Dataset_FTTT_2026` folder extracted next to this repository. The dataset is intentionally not bundled into the submission ZIP.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Add your real `GROQ_API_KEY` to `.env`. Add `GNANI_API_KEY` after creating an account at https://app.gnani.ai/voice. Never commit `.env`.

If the dataset is in a different location, update `DATASET_DIR` in `.env`.

## Run a text question (Steps 1-3)

```bash
python run.py --text "How fast are refunds to the ZentroMart Wallet?"
```

### Run Multi-turn Conversation Evaluation (23 Scenarios, 96 Turns)

```bash
python evaluate_conversations.py
```
This tests all 23 multi-turn support dialogues in `practice/conversations.csv`, verifying caller verification gates (`lookup_order`), human escalation tickets (`create_ticket`), date arithmetic for returns/refunds, and decision classification (`100% Turn Pass`). Results are written to `sample_outputs/conversation_evaluation.csv`.

### Run Single-Question Benchmark (Baseline vs RAG)

```bash
python evaluate.py
```
This compares the prompting-only LLM baseline (Step 1) against our section-aware RAG agent (Step 2) across all 52 practice questions. Results are saved to `sample_outputs/evaluation.csv`.

## Run voice batch (Step 4)

Put customer audio files (`.wav`, `.mp3`, `.amr`) in a folder, then run:

```bash
python run.py --audio-dir ./customer_audio --output-dir ./sample_outputs --language en-IN
```

Each input produces a `.json` file containing the transcript, answer, retrieved sources and audio path, plus a synthesized `*_reply.wav`. Use `hi-IN` for Hindi. Gnani supports `en-IN` for English mixed with Hindi. No audio files are included in the supplied dataset, so record your own voice for testing.

## Dataset and policy rules

- The dataset's reference date is **20 October 2026**, not the machine's current date.
- CSV IDs and pincodes must remain strings.
- Current policy outranks agent guides, help, FAQs, and marketing pages. Superseded policy is not authoritative.
- Order details must only be disclosed after `lookup_order` verifies the caller's registered phone suffix or delivery pincode.
- Tickets are only confirmed when `create_ticket` returns `created=True`.
- The provided data is synthetic and must remain so.

## Current limitations to disclose

- Retrieval uses a dependency-light BM25 index, not embeddings. It preserves section IDs and downranks superseded policy; inspect sources and test paraphrases.
- Tool routing is intentionally conservative and should be validated against all 23 practice conversations. It is a capstone prototype, not production authentication.
- The included evaluator reports a lexical heuristic score, not an official capstone score.
- Voice mode is fully operational with live Gnani.ai STT (`en-IN` / `hi-IN`) and TTS (`timbre-v2.5` Kaveri voice).

## Submission files

Include your public repository URL in the submission package after publishing it:
`https://github.com/JAIDUUU/anomus-zentromart`

Also include `.env.example`, `requirements.txt`, a run command, a report (maximum 4 pages), a demo video (maximum 2 minutes), and sample outputs. Do not include API keys, virtual environments, or a duplicate dataset copy.
