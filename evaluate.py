"""Evaluate prompting-only vs RAG on practice/questions.csv; writes transparent CSV scores.

Strictly follows DATA_DICTIONARY.md specifications:
- ' ; ' separates independent required facts (AND logic).
- ' / ' separates alternative phrasings for a single fact (OR logic).
- 'must_not_include' separates forbidden claims.
- Compares Step 1 (Prompt-only baseline) vs Step 2 (RAG) on all 52 practice questions.
"""
import csv, re, os, time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from src.agent import SupportAgent, SYSTEM_PROMPT, parse_retry_after
from src.config import DATASET_DIR, GROQ_MODEL

load_dotenv()

def norm(s):
    return re.sub(r'\s+', ' ', (s or '').lower()).strip()

def check_must_include(text, must_include_str):
    if not must_include_str or not must_include_str.strip():
        return True
    text_norm = norm(text)
    facts = [f.strip() for f in must_include_str.split(';') if f.strip()]
    for fact in facts:
        alts = [a.strip().lower() for a in fact.split('/') if a.strip()]
        if not any(alt in text_norm for alt in alts):
            return False
    return True

def check_must_not_include(text, must_not_include_str):
    if not must_not_include_str or not must_not_include_str.strip():
        return True
    text_norm = norm(text)
    for term in re.split(r'[;/]', must_not_include_str):
        term = term.strip().lower()
        if term and term in text_norm:
            return False
    return True

def call_groq_baseline(client, q):
    for attempt in range(12):
        try:
            res = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.1,
                max_tokens=140,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT + "\nNote: You have NO company documents or order records available. Answer as best you can."},
                    {'role': 'user', 'content': q}
                ]
            )
            return res.choices[0].message.content or ""
        except RateLimitError as e:
            wait = parse_retry_after(str(e))
            time.sleep(wait)
        except Exception:
            time.sleep(3.0)
    return "Sorry, I do not have that information."

def main():
    src = DATASET_DIR / 'practice' / 'questions.csv'
    rows = list(csv.DictReader(src.open(encoding='utf-8-sig', newline='')))
    target = (Path(__file__).resolve().parent / 'sample_outputs' / 'evaluation.csv').resolve()

    # Load existing baseline and valid RAG cache if present
    cached_baseline = {}
    cached_rag = {}
    if target.exists():
        try:
            with target.open(encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    qid_val = r.get('id')
                    if qid_val and r.get('prompt_only_answer') and 'temporarily unable' not in r.get('prompt_only_answer', ''):
                        cached_baseline[qid_val] = r['prompt_only_answer']
                    if qid_val and r.get('rag_answer') and r.get('rag_heuristic_pass') == '1':
                        cached_rag[qid_val] = r
        except Exception:
            pass

    client = Groq()
    agent = SupportAgent()
    out = []

    print(f"Starting evaluation of {len(rows)} practice questions (Prompt-Only vs RAG)...")

    gold_recall_hits = 0
    gold_total_count = 0

    for i, row in enumerate(rows, 1):
        q = row.get('question', '')
        qid = row.get('id', f'PQ{i}')
        gold = row.get('gold_evidence', '').strip()

        # Step 1: Prompt-only baseline (re-use cache or call)
        if qid in cached_baseline and cached_baseline[qid].strip():
            base = cached_baseline[qid]
        else:
            base = call_groq_baseline(client, q)
            time.sleep(1.5)

        # Step 2: RAG Agent
        rag_res = agent.respond(q, session_id=f'eval-q-{i}')
        rag_ans = rag_res['answer']
        rag_sources = rag_res.get('sources', [])

        must_inc = row.get('must_include', '')
        must_not = row.get('must_not_include', '')

        base_pass = int(check_must_include(base, must_inc) and check_must_not_include(base, must_not))
        rag_pass = int(check_must_include(rag_ans, must_inc) and check_must_not_include(rag_ans, must_not))

        # Check gold evidence recall
        if gold:
            gold_total_count += 1
            retrieved_ids = {s['section_id'] for s in rag_sources}
            # All required gold sections in gold string (separated by ;)
            gold_sections = [g.strip() for g in gold.split(';') if g.strip()]
            if all(any(alt.strip() in retrieved_ids for alt in gsec.split('/')) for gsec in gold_sections):
                gold_recall_hits += 1

        out.append({
            'id': qid,
            'language': row.get('language', ''),
            'question': q,
            'expected_decision': row.get('expected_decision'),
            'gold_evidence': gold,
            'prompt_only_answer': base,
            'rag_answer': rag_ans,
            'prompt_only_heuristic_pass': base_pass,
            'rag_heuristic_pass': rag_pass,
            'rag_sources': ';'.join(x['section_id'] for x in rag_sources)
        })

        status = f"Baseline: {'PASS' if base_pass else 'FAIL'} | RAG: {'PASS' if rag_pass else 'FAIL'}"
        print(f"[{i:02d}/{len(rows)}] {qid}: {status}")

    target.parent.mkdir(exist_ok=True)
    with target.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=out[0].keys())
        w.writeheader()
        w.writerows(out)

    base_total = sum(r['prompt_only_heuristic_pass'] for r in out)
    rag_total = sum(r['rag_heuristic_pass'] for r in out)

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Total Questions Evaluated: {len(out)}")
    print(f"Step 1 (Prompt-only Baseline): {base_total}/{len(out)} ({base_total/len(out)*100:.1f}%)")
    print(f"Step 2 (Grounded RAG Agent) : {rag_total}/{len(out)} ({rag_total/len(out)*100:.1f}%)")
    print(f"Gold Section Retrieval Recall: {gold_recall_hits}/{gold_total_count} ({gold_recall_hits/max(1,gold_total_count)*100:.1f}%)")
    print(f"Improvement over baseline   : +{rag_total - base_total} (+{(rag_total-base_total)/len(out)*100:.1f}%)")
    print(f"Saved evaluation results to : {target}")

if __name__ == '__main__':
    main()
