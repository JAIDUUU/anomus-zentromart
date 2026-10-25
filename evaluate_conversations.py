"""Evaluate agent on practice/conversations.csv across all 23 multi-turn scenarios."""
import csv, re, os
from pathlib import Path
from dotenv import load_dotenv
from src.agent import SupportAgent
from src.config import DATASET_DIR

load_dotenv()

def norm(s):
    return re.sub(r'\s+', ' ', (s or '').lower()).strip()

def check_must_include(text, must_include_str):
    if not must_include_str or not must_include_str.strip():
        return True
    text_norm = norm(text)
    # Required facts separated by ' ; '
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

def main():
    csv_path = DATASET_DIR / 'practice' / 'conversations.csv'
    with csv_path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))

    agent = SupportAgent()
    results = []

    # Group by conversation
    conv_groups = {}
    for r in rows:
        cid = r['conversation_id']
        conv_groups.setdefault(cid, []).append(r)

    total_turns = len(rows)
    passed_turns = 0
    tool_matched_turns = 0

    print(f"Starting multi-turn conversation evaluation: {len(conv_groups)} conversations, {total_turns} turns total.")

    for cid, turns in sorted(conv_groups.items()):
        session_id = f"eval-conv-{cid}"
        print(f"\n--- Running Conversation {cid} ({len(turns)} turns) ---")
        for r in sorted(turns, key=lambda x: int(x['turn'])):
            turn_no = r['turn']
            user_msg = r['customer_message']
            exp_dec = r.get('expected_decision', '')
            exp_tools = r.get('expected_tool_calls', '')
            must_inc = r.get('must_include', '')
            must_not = r.get('must_not_include', '')

            out = agent.respond(user_msg, session_id=session_id)
            reply = out['answer']
            calls = out.get('tool_calls', [])

            inc_pass = check_must_include(reply, must_inc)
            exc_pass = check_must_not_include(reply, must_not)
            text_pass = inc_pass and exc_pass

            # Tool matching check
            tool_pass = True
            if exp_tools:
                for exp_t in [t.strip() for t in exp_tools.split(';') if t.strip()]:
                    func_name = exp_t.split('(')[0]
                    if not any(func_name in c for c in calls):
                        tool_pass = False

            if text_pass:
                passed_turns += 1
            if tool_pass:
                tool_matched_turns += 1

            status = "PASS" if text_pass and tool_pass else "FAIL"
            print(f"  T{turn_no} [{status}] User: {user_msg[:40]} -> ExpDec: {exp_dec} | Tools: {calls}")

            results.append({
                'conversation_id': cid,
                'turn': turn_no,
                'scenario': r.get('scenario', ''),
                'customer_message': user_msg,
                'agent_reply': reply,
                'expected_decision': exp_dec,
                'agent_decision': out.get('decision', ''),
                'expected_tools': exp_tools,
                'agent_tool_calls': ';'.join(calls),
                'must_include_pass': int(inc_pass),
                'must_not_include_pass': int(exc_pass),
                'tool_pass': int(tool_pass),
                'turn_pass': int(text_pass and tool_pass)
            })

    out_file = Path('sample_outputs/conversation_evaluation.csv')
    out_file.parent.mkdir(exist_ok=True)
    with out_file.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Total Turns Evaluated: {total_turns}")
    print(f"Text Heuristic Pass: {passed_turns}/{total_turns} ({passed_turns/total_turns*100:.1f}%)")
    print(f"Tool Match Pass: {tool_matched_turns}/{total_turns} ({tool_matched_turns/total_turns*100:.1f}%)")
    overall_pass = sum(r['turn_pass'] for r in results)
    print(f"Overall Turn Pass: {overall_pass}/{total_turns} ({overall_pass/total_turns*100:.1f}%)")
    print(f"Saved results to: {out_file}")

if __name__ == '__main__':
    main()
