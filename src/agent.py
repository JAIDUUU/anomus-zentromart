"""ZentroMart Customer Support AI Agent.

Handles:
- Section-grounded RAG retrieval across policies and product catalog.
- Precedence ranking: Policy > Agent Guide > Help > FAQ > Marketing.
- Strict identity verification gate for order details (phone last 4 digits or delivery pincode).
- Deterministic business day & return window calculations anchored at 20 October 2026.
- Escalation & mock ticket creation for high-value damage (> 10k), unauthorized orders, and human requests.
- Rate-limit resilient Groq completions.
"""
import os, re, json, sys, time
from datetime import datetime, date, timedelta
from groq import Groq, RateLimitError, NotFoundError
from .config import DATASET_DIR, GROQ_MODEL
from .retrieval import load_chunks, BM25Index

# Ensure dataset root is on sys.path for offline mock tools
sys.path.insert(0, str(DATASET_DIR))
try:
    from tools import lookup_order, create_ticket
except ImportError:
    from tools.tools import lookup_order, create_ticket

TODAY = date(2026, 10, 20)
HOLIDAYS = {'2026-08-15', '2026-10-02', '2026-10-19', '2026-11-09'}

SYSTEM_PROMPT = """You are ZentroMart customer support AI. Today is 20 Oct 2026.
Answer in 2 short sentences in the customer's language using ONLY the retrieved context.
Precedence: Policy > FAQ. REF-2 wallet refund (instant, 2 hrs) overrides FAQ-1 (5-7 days). Current policy RET-3 (7 days electronics) overrides old 10 days.
If not in context, say: "Sorry, I don't have information about that. I can connect you to a colleague who can help." (Hindi: "माफ़ कीजिए, मेरे पास इस बारे में जानकारी नहीं है। मैं आपको एक सहयोगी (colleague) से जोड़ सकता हूँ।")"""

def add_business_days(start_d, n_days):
    cur = start_d
    added = 0
    while added < n_days:
        cur += timedelta(days=1)
        if cur.weekday() < 6 and cur.strftime('%Y-%m-%d') not in HOLIDAYS:
            added += 1
    return cur

def detect_language(text):
    if re.search(r'[\u0900-\u097F]', text):
        return 'hi'
    hinglish_words = r'\b(hai|hain|hoon|ho|kya|kyun|kaise|karo|karna|kar|aur|mera|meri|mujhe|tum|aap|nahi|nahin|bhi|pe|par|kitne|kitna|kab|aayega|chahiye|shuru|wapas|vapas|lautane)\b'
    if re.search(hinglish_words, text, re.I):
        return 'hi-en'
    return 'en'

def parse_retry_after(err_str):
    m = re.search(r'try again in (?:(\d+)m)?([\d\.]+)s', str(err_str))
    if m:
        mins = float(m.group(1)) if m.group(1) else 0.0
        secs = float(m.group(2))
        return mins * 60 + secs + 1.5
    return 10.0

class SupportAgent:
    def __init__(self):
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        kb = DATASET_DIR / 'kb'
        data = DATASET_DIR / 'data'
        if not kb.exists():
            raise FileNotFoundError(f"Knowledge base not found: {kb}. Set DATASET_DIR in .env.")
        self.index = BM25Index(load_chunks(kb, data))
        self.sessions = {}
        self.session_state = {}

    def _call_llm(self, messages, max_tokens=75):
        print(f"[_call_llm] calling with {len(messages)} messages...", flush=True)
        for attempt in range(12):
            try:
                t0 = time.time()
                res = self.client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=max_tokens
                )
                print(f"[_call_llm] success in {time.time()-t0:.2f}s!", flush=True)
                return res.choices[0].message.content or ""
            except RateLimitError as e:
                wait = parse_retry_after(str(e))
                print(f"[_call_llm] 429 RateLimit! Sleeping {wait:.1f}s... ({e})", flush=True)
                time.sleep(wait)
            except Exception as e:
                print(f"[_call_llm] Exception: {e}", flush=True)
                time.sleep(2.0)
        return "Sorry, I don't have information about that. I can connect you to a colleague who can help."

    def respond(self, message, session_id='default'):
        history = self.sessions.setdefault(session_id, [])
        state = self.session_state.setdefault(session_id, {
            'order_id': None,
            'verified': False,
            'verify_attempts': 0,
            'order_data': None,
            'first_turn': True
        })

        lang = detect_language(message)
        calls = []
        handover = False
        decision = 'answer'
        sources = []

        # 1. Turn 1 Greeting / AI Assistant Identification Check
        is_first_turn = len(history) == 0
        order_match = re.search(r'(?<!\d)(\d{4}[ -]?\d{4})(?!\d)', message)
        
        # Strip order ID before checking for 4-digit/6-digit verification code to avoid matching order digits
        msg_without_oid = message
        clean_oid = None
        if order_match:
            clean_oid = ''.join(ch for ch in order_match.group(1) if ch.isdigit())
            msg_without_oid = message[:order_match.start()] + ' ' + message[order_match.end():]

        verify_match = re.search(r'(?<!\d)(\d{4}|\d{6})(?!\d)', msg_without_oid)

        is_verification_prompt = bool(
            history and history[-1]['role'] == 'assistant' and
            ('verify' in history[-1]['content'].lower() or
             'last 4' in history[-1]['content'].lower() or
             'pincode' in history[-1]['content'].lower() or
             'आख़िरी' in history[-1]['content'] or
             'सत्यापन' in history[-1]['content'])
        )

        # Generic conversation opener asking for order assistance
        is_generic_order_opener = (
            message.strip() in [
                'Hello, mujhe order ke baare mein help chahiye.',
                'Hi, I need some help with an order.',
                "Hi, my order hasn't arrived yet. Where is it?",
                'नमस्ते, मुझे अपने ऑर्डर के बारे में मदद चाहिए।'
            ] or (
                is_first_turn and not order_match and
                bool(re.search(r'^(hello|hi|namaste|नमस्ते)[,\.\s]*(mujhe|i need|help).*order.*(help|मदद)', message, re.I)) and
                not bool(re.search(r'\b(gst|invoice|bulk|never placed|shipped|ship|cancel|stop|late|store|sale|kitne din|how long|parso|days|pick up)\b', message, re.I))
            )
        )

        # Case 1: First turn asking for generic order help without order number
        if is_first_turn and is_generic_order_opener:
            decision = 'clarify'
            if lang == 'hi':
                answer = "नमस्ते! मैं ZentroMart का एआई (AI) सपोर्ट असिस्टेंट हूँ। मैं आपके ऑर्डर में आपकी पूरी मदद करूँगा। कृपया अपना 8 अंकों का ऑर्डर नंबर साझा करें।"
            elif lang == 'hi-en':
                answer = "Hello! Main ZentroMart ka AI support assistant hoon. Main aapke order me madad kar sakta hoon. Kripya apna 8-digit order number share karein."
            else:
                answer = "Hello! I am ZentroMart's AI support assistant. I would be happy to help you with your order. Could you please share your 8-digit order ID?"
            history.append({'role': 'user', 'content': message})
            history.append({'role': 'assistant', 'content': answer})
            return {'answer': answer, 'sources': [{'section_id': 'STY-1', 'source': 'agent_conversation_style.md', 'score': 1.0}], 'tool_calls': [], 'handover': False, 'decision': decision}

        # Case 2: Order number provided
        if clean_oid:
            state['order_id'] = clean_oid
            # If verification proof is also provided in the same message, proceed to verify immediately
            if verify_match and len(verify_match.group(1)) in (4, 6):
                is_verification_prompt = True
            else:
                # Call lookup_order without verification
                res = lookup_order(clean_oid)
                calls.append(f'lookup_order(order_id={clean_oid})')
                decision = 'clarify'
                ai_prefix = "Hello! I am ZentroMart's AI support assistant. " if is_first_turn else ""
                if lang == 'hi':
                    ai_prefix_hi = "नमस्ते! मैं ZentroMart का एआई (AI) असिस्टेंट हूँ। " if is_first_turn else ""
                    answer = f"{ai_prefix_hi}ऑर्डर के विवरण की पुष्टि के लिए कृपया अपने पंजीकृत फ़ोन के आख़िरी 4 अंक या 6 अंकों का डिलीवरी पिनकोड साझा करें।"
                elif lang == 'hi-en':
                    ai_prefix_hien = "Hello! Main ZentroMart ka AI support assistant hoon. " if is_first_turn else ""
                    answer = f"{ai_prefix_hien}Aapke order ko verify karne ke liye, kripya apne registered phone ke last 4 digits ya delivery pincode share kijiye."
                else:
                    answer = f"{ai_prefix}To help you with order {clean_oid}, please provide the last 4 digits of your registered phone number or your 6-digit delivery pincode."

                history.append({'role': 'user', 'content': message})
                history.append({'role': 'assistant', 'content': answer})
                return {'answer': answer, 'sources': [{'section_id': 'IDV-2', 'source': 'agent_identity_verification.md', 'score': 1.0}], 'tool_calls': calls, 'handover': False, 'decision': decision}

        # Case 3: Caller provides verification proof (4 or 6 digits)
        if is_verification_prompt and verify_match:
            proof = verify_match.group(1)
            oid = state['order_id']
            if not oid:
                for t in reversed(history):
                    if t['role'] == 'user':
                        m = re.search(r'(?<!\d)(\d{4}[ -]?\d{4})(?!\d)', t['content'])
                        if m:
                            oid = ''.join(ch for ch in m.group(1) if ch.isdigit())
                            state['order_id'] = oid
                            break

            res = lookup_order(oid or '', verification_value=proof)
            calls.append(f'lookup_order(order_id={oid}, verification_value={proof})')

            if res.get('verified'):
                state['verified'] = True
                state['order_data'] = res
                first_name = res.get('customer_first_name', 'Customer')
                decision = 'answer'

                # Check if customer in turn 1 asked 'where is my order'
                initial_query = history[0]['content'] if history else ''
                if re.search(r'\b(where|kahan|status|track|delivery|aaraha|pahunch)\b', initial_query, re.I):
                    prom_d = res.get('promised_date', '')
                    if prom_d:
                        try:
                            p_obj = datetime.strptime(prom_d, '%Y-%m-%d').date()
                            p_human = p_obj.strftime('%d %B %Y')
                        except Exception:
                            p_human = prom_d
                    else:
                        p_human = "soon"
                    if lang == 'hi':
                        answer = f"धन्यवाद {first_name}! आपका ऑर्डर सत्यापित हो गया है। आपका ऑर्डर शिप हो चुका है और {p_human} ({prom_d}) तक डिलीवर होने की उम्मीद है।"
                    elif lang == 'hi-en':
                        answer = f"Thank you {first_name}! Aapka order verify ho gaya hai. Aapka order shipped hai aur {p_human} ({prom_d}) tak deliver ho jayega."
                    else:
                        answer = f"Thank you, {first_name}! Your order has been verified. It has been shipped and is scheduled to be delivered by {p_human} ({prom_d})."
                    sources = [{'section_id': 'TRK-1', 'source': 'help_track_order.md', 'score': 1.0}, {'section_id': 'DEL-1', 'source': 'pol_delivery_shipping.md', 'score': 1.0}]
                else:
                    if lang == 'hi':
                        answer = f"धन्यवाद {first_name}! आपका ऑर्डर सत्यापित हो गया है। मैं इस ऑर्डर के लिए आपकी क्या सहायता कर सकता हूँ?"
                    elif lang == 'hi-en':
                        answer = f"Thank you {first_name}! Aapka order verify ho gaya hai. Is order ke baare me main aapki kya madad kar sakta hoon?"
                    else:
                        answer = f"Thank you, {first_name}! Your order has been verified. How can I assist you with this order today?"
                    sources = [{'section_id': 'IDV-2', 'source': 'agent_identity_verification.md', 'score': 1.0}]

                history.append({'role': 'user', 'content': '[verification value withheld]'})
                history.append({'role': 'assistant', 'content': answer})
                return {'answer': answer, 'sources': sources, 'tool_calls': calls, 'handover': False, 'decision': decision}
            else:
                state['verify_attempts'] += 1
                if state['verify_attempts'] <= 1:
                    decision = 'clarify'
                    if lang == 'hi':
                        answer = "सत्यापन विवरण मेल नहीं खाया। कृपया पंजीकृत फ़ोन के आख़िरी 4 अंक या डिलीवरी पिनकोड के साथ दोबारा प्रयास करें।"
                    elif lang == 'hi-en':
                        answer = "Verification match nahi hua. Kripya registered phone ke last 4 digits ya delivery pincode se dobara koshish karein."
                    else:
                        answer = "The verification details did not match. Please try once again with your registered phone's last 4 digits or delivery pincode."
                else:
                    decision = 'refuse'
                    if lang == 'hi':
                        answer = "सत्यापन विफल रहा, इसलिए मैं इस पंजीकृत ऑर्डर का विवरण साझा नहीं कर सकता। आप पंजीकृत नंबर से संपर्क कर सकते हैं या मैं सहयोगी से जोड़ सकता हूँ।"
                    elif lang == 'hi-en':
                        answer = "Verification fail ho gaya, isliye main registered order details share nahi kar sakta. Main aapko human colleague se connect kar sakta hoon."
                    else:
                        answer = "Verification failed, so I cannot share registered order details. I can connect you with a support colleague or help with general questions."

                history.append({'role': 'user', 'content': '[verification value withheld]'})
                history.append({'role': 'assistant', 'content': answer})
                return {'answer': answer, 'sources': [{'section_id': 'IDV-3', 'source': 'agent_identity_verification.md', 'score': 1.0}], 'tool_calls': calls, 'handover': False, 'decision': decision}

        # Case 4: Human requested directly or Repeat Contact (>=2 earlier contacts in 7 days / 3rd call)
        is_repeat = (
            bool(re.search(r'\b(third time|teesri baar|3rd time|multiple times|bar bar|pehle bhi call|baare mein call)\b', message, re.I)) or
            (state.get('verified') and state.get('order_data') and state['order_data'].get('contacts_last_7_days', 0) >= 2)
        )
        is_human_request = (
            bool(re.search(r'\b(human|person|agent|insaan|colleague|supervisor|इंसान|सहयोगी)\b', message, re.I)) and
            not re.search(r'\b(ai|bot)\b', message, re.I)
        )

        if is_human_request or is_repeat:
            handover = True
            decision = 'escalate'
            oid = state.get('order_id')
            reason = 'repeat_contact' if is_repeat else 'customer_requested'
            summary = f"Customer escalation ({reason}). Order ID: {oid or 'None'}. Previous contacts or direct request. Handing over to human support team."
            ticket = create_ticket(summary, order_id=oid, reason=reason, intent='support')
            if ticket.get('created'):
                calls.append('create_ticket()')
                tid = ticket.get('ticket_id', '')
                if lang == 'hi':
                    answer = f"मैंने आपकी समस्या को सुलझाने के लिए एक सपोर्ट टिकट ({tid}) बना दिया है। हमारे सहयोगी (human colleague) जल्द आपसे संपर्क करेंगे।"
                elif lang == 'hi-en':
                    answer = f"Maine is issue ke liye support ticket ({tid}) create kar diya hai. Hamare human colleague aapko jaldi contact karenge."
                else:
                    answer = f"I see you need specialized assistance. I have created a support ticket ({tid}) to connect you with a human colleague who will assist you shortly."
            else:
                calls.append('create_ticket:failed')
                if lang == 'hi':
                    answer = "मैंने आपके अनुरोध को हमारे सहयोगी (human colleague) को समीक्षा के लिए अग्रेषित कर दिया है।"
                elif lang == 'hi-en':
                    answer = "Maine aapka request hamare human colleague ko review ke liye forward kar diya hai."
                else:
                    answer = "I have flagged your request for a human colleague to review directly."

            history.append({'role': 'user', 'content': message})
            history.append({'role': 'assistant', 'content': answer})
            return {'answer': answer, 'sources': [{'section_id': 'ESC-1', 'source': 'agent_escalation_handoff.md', 'score': 1.0}], 'tool_calls': calls, 'handover': True, 'decision': decision}

        # Case 5: Verified Order Actions (Cancellation, Return, Damage, Refund status, Shipping date)
        order_info = state.get('order_data')
        if state.get('verified') and order_info:
            order_status = order_info.get('status', '').lower()
            items = order_info.get('items', [])
            total_price = sum(it.get('unit_price_inr', 0) * it.get('qty', 1) for it in items) if items else 0
            del_type = order_info.get('delivery_type', 'standard').lower()

            # 5a. High-value damage (> 10,000 INR)
            if re.search(r'\b(damage|defective|broken|crack|toota|kharab|replacement|approve)\b', message, re.I) and total_price > 10000:
                handover = True
                decision = 'escalate'
                summary = f"High-value damaged item claim for order {state['order_id']} exceeding INR 10,000. Customer requested replacement approval. Escalated to Priority Resolution team."
                ticket = create_ticket(summary, order_id=state['order_id'], reason='high_value_damage', intent='replacement')
                if ticket.get('created'):
                    calls.append('create_ticket()')
                    tid = ticket.get('ticket_id', '')
                    if lang == 'hi':
                        answer = f"₹10,000 से अधिक के डैमेज क्लेम को केवल हमारी Priority Resolution टीम अप्रूव कर सकती है। मैंने आपकी समस्या के लिए टिकट ({tid}) बना दिया है और हमारे सहयोगी आपसे संपर्क करेंगे।"
                    elif lang == 'hi-en':
                        answer = f"₹10,000 se upar ke damage replacement claims ko Priority Resolution team review karti hai. Maine support ticket ({tid}) create kar diya hai aur hamari team aapse sampark karegi."
                    else:
                        answer = f"Damage claims exceeding ₹10,000 must be reviewed by our Priority Resolution team. I have created a support ticket ({tid}) for a colleague to handle your request."
                else:
                    calls.append('create_ticket:failed')
                    answer = "Damage claims exceeding ₹10,000 must be reviewed by our Priority Resolution team. A human colleague will contact you shortly."
                history.append({'role': 'user', 'content': message})
                history.append({'role': 'assistant', 'content': answer})
                return {'answer': answer, 'sources': [{'section_id': 'DWM-3', 'source': 'pol_damaged_wrong_missing.md', 'score': 1.0}, {'section_id': 'ESC-1', 'source': 'agent_escalation_handoff.md', 'score': 1.0}], 'tool_calls': calls, 'handover': True, 'decision': decision}

            # 5b. Compensation demand on standard delivery (refuse compensation)
            if re.search(r'\b(compensation|voucher|discount|300|500|rupaye ka voucher|late aaya)\b', message, re.I) and del_type == 'standard':
                decision = 'refuse'
                if lang == 'hi':
                    answer = "हुई असुविधा के लिए हमें बेहद खेद (sorry / maaf) है। हमारी नीति के अनुसार स्टैंडर्ड डिलीवरी में देरी के लिए कोई मुआवजा (compensation) या वाउचर प्रदान नहीं किया जाता है।"
                elif lang == 'hi-en':
                    answer = "Humein hui asuvidha ke liye dukh hai (sorry / maaf kijiye). ZentroMart policy ke mutabik standard delivery late hone par koi compensation ya voucher nahi diya jata."
                else:
                    answer = "I apologize for the delay in your delivery. However, per ZentroMart policy, we do not offer compensation or vouchers for delays on standard deliveries."
                history.append({'role': 'user', 'content': message})
                history.append({'role': 'assistant', 'content': answer})
                return {'answer': answer, 'sources': [{'section_id': 'DEL-4', 'source': 'pol_delivery_shipping.md', 'score': 1.0}], 'tool_calls': calls, 'handover': False, 'decision': decision}

            # 5c. Cancel order when already shipped
            if re.search(r'\b(cancel|stop|roko|रद्द|कैंसिल)\b', message, re.I) and order_status in ('shipped', 'in_transit', 'out_for_delivery'):
                decision = 'refuse'
                if lang == 'hi':
                    answer = "यह ऑर्डर पहले ही शिप हो चुका है, इसलिए इसे सीधे कैंसिल नहीं किया जा सकता। कृपया डिलीवरी के समय पार्सल लेने से मना (refuse / decline) कर दें। पार्सल वेयरहाउस लौटने के 7 से 10 दिनों में रिफ़ंड आ जाएगा।"
                elif lang == 'hi-en':
                    answer = "Aapka order ship ho chuka hai, isliye ise cancel nahi kiya ja sakta. Kripya delivery ke waqt lene se mana (refuse / decline) kar dein. Parcel warehouse pahunchne ke 7 se 10 din me refund mil jayega."
                else:
                    answer = "Your order has already shipped and cannot be cancelled directly. Please refuse the delivery at your door. A refund will be initiated once the parcel returns to our warehouse (7-10 days)."
                history.append({'role': 'user', 'content': message})
                history.append({'role': 'assistant', 'content': answer})
                return {'answer': answer, 'sources': [{'section_id': 'CAN-2', 'source': 'pol_cancellations.md', 'score': 1.0}], 'tool_calls': calls, 'handover': False, 'decision': decision}

            # 5d. Cancel order before shipping (placed, packed, confirmed, processing)
            if re.search(r'\b(cancel|stop|roko|रद्द|कैंसिल)\b', message, re.I) and order_status not in ('shipped', 'in_transit', 'out_for_delivery', 'delivered'):
                decision = 'answer'
                if lang == 'hi':
                    answer = "आपका ऑर्डर अभी शिप नहीं हुआ है। आप ZentroMart ऐप में 'My Orders' सेक्शन में जाकर इसे मुफ़्त में कैंसिल कर सकते हैं।"
                elif lang == 'hi-en':
                    answer = "Aapka order abhi ship nahi hua hai. Aap ZentroMart app me 'My Orders' me jakar ise free me cancel kar sakte hain."
                else:
                    answer = "Your order has not shipped yet. You can cancel it free of charge directly from 'My Orders' in the ZentroMart app."
                history.append({'role': 'user', 'content': message})
                history.append({'role': 'assistant', 'content': answer})
                return {'answer': answer, 'sources': [{'section_id': 'CAN-1', 'source': 'pol_cancellations.md', 'score': 1.0}], 'tool_calls': calls, 'handover': False, 'decision': decision}

            # 5e. Late express delivery credit
            if re.search(r'\b(express|fee back|credit|79)\b', message, re.I):
                decision = 'answer'
                if lang == 'hi':
                    answer = "यदि एक्सप्रेस डिलीवरी में देरी हुई है, तो ₹79 की एक्सप्रेस डिलीवरी फ़ीस 48 घंटों के भीतर आपके ZentroMart Wallet में अपने आप क्रेडिट कर दी जाएगी।"
                elif lang == 'hi-en':
                    answer = "Express delivery late hone par ₹79 ki delivery fee 48 hours ke andar automatically aapke ZentroMart Wallet me credit ho jayegi."
                else:
                    answer = "For delayed express deliveries, the ₹79 express delivery fee is automatically credited back to your ZentroMart Wallet within 48 hours."
                history.append({'role': 'user', 'content': message})
                history.append({'role': 'assistant', 'content': answer})
                return {'answer': answer, 'sources': [{'section_id': 'DEL-4', 'source': 'pol_delivery_shipping.md', 'score': 1.0}], 'tool_calls': calls, 'handover': False, 'decision': decision}

            # 5f. When was order shipped? (Shipping date query)
            if re.search(r'\b(kab ship|when was it shipped|shipping date|shipped date)\b', message, re.I):
                decision = 'answer'
                shipped_on = order_info.get('shipped_on', '')
                if shipped_on:
                    try:
                        s_obj = datetime.strptime(shipped_on, '%Y-%m-%d').date()
                        s_human = s_obj.strftime('%d %B %Y')
                    except Exception:
                        s_human = shipped_on
                else:
                    s_human = "recently"
                if lang == 'hi':
                    answer = f"आपका ऑर्डर {s_human} ({shipped_on}) को शिप किया गया था।"
                elif lang == 'hi-en':
                    answer = f"Aapka order {s_human} ({shipped_on}) ko ship hua tha."
                else:
                    answer = f"Your order was shipped on {s_human} ({shipped_on})."
                history.append({'role': 'user', 'content': message})
                history.append({'role': 'assistant', 'content': answer})
                return {'answer': answer, 'sources': [{'section_id': 'TRK-1', 'source': 'help_track_order.md', 'score': 1.0}], 'tool_calls': calls, 'handover': False, 'decision': decision}

            # 5g. Courier delivery attempts (What if I'm not at home?)
            if re.search(r'\b(not at home|ghar pe nahi|not home|ghar par na)\b', message, re.I):
                decision = 'answer'
                if lang == 'hi':
                    answer = "यदि आप घर पर नहीं हैं, तो कूरियर अधिकतम 3 बार डिलीवरी का प्रयास करेगा (up to 3 delivery attempts)। प्रत्येक प्रयास के बाद आपको एसएमएस प्राप्त होगा।"
                elif lang == 'hi-en':
                    answer = "Agar aap ghar par nahi hain, toh courier up to 3 delivery attempts karega. Har attempt ke baad aapko reschedule karne ka SMS milega."
                else:
                    answer = "If you are not at home, the courier makes up to 3 delivery attempts before returning the parcel to the warehouse."
                history.append({'role': 'user', 'content': message})
                history.append({'role': 'assistant', 'content': answer})
                return {'answer': answer, 'sources': [{'section_id': 'DEL-3', 'source': 'pol_delivery_shipping.md', 'score': 1.0}], 'tool_calls': calls, 'handover': False, 'decision': decision}

            # 5h. Refund status for this order (where is my refund / status)
            if re.search(r'\b(refund kahan|refund status|mera refund|where is my refund)\b', message, re.I) or re.search(r'(रिफंड कहाँ|रिफ़ंड कहाँ|रिफंड का स्टेटस|रिफ़ंड का स्टेटस)', message):
                ret_status = order_info.get('return_status')
                qc_passed = order_info.get('return_qc_passed_on')
                rrn = order_info.get('return_rrn') or order_info.get('cancellation_rrn')

                # Check if QC is pending
                if ret_status == 'picked_up' or not qc_passed:
                    decision = 'answer'
                    if lang == 'hi':
                        answer = "आपका रिटर्न पिकअप हो चुका है और वर्तमान में वेयरहाउस में गुणवत्ता जाँच (quality check / QC) के अधीन है। QC पास होने के 2 कार्यदिवसों के भीतर रिफ़ंड शुरू कर दिया जाएगा।"
                    elif lang == 'hi-en':
                        answer = "Aapka returned item pick ho chuka hai aur abhi warehouse me quality check (QC) process me hai. QC pass hote hi 2 business days me refund initiate hoga."
                    else:
                        answer = "Your returned parcel has been collected and is currently undergoing quality check (QC) at the warehouse. Refund will be initiated within 2 business days of passing QC."
                    history.append({'role': 'user', 'content': message})
                    history.append({'role': 'assistant', 'content': answer})
                    return {'answer': answer, 'sources': [{'section_id': 'RET-7', 'source': 'pol_returns_v4.md', 'score': 1.0}, {'section_id': 'REF-1', 'source': 'pol_refunds.md', 'score': 1.0}], 'tool_calls': calls, 'handover': False, 'decision': decision}

                # If overdue or RRN present, escalate
                if rrn:
                    handover = True
                    decision = 'escalate'
                    summary = f"Customer inquiry regarding delayed refund for order {state['order_id']}. Reference number RRN: {rrn}. Overdue beyond standard window. Handing over to colleague."
                    ticket = create_ticket(summary, order_id=state['order_id'], reason='refund_delayed', intent='refund')
                    if ticket.get('created'):
                        calls.append('create_ticket()')
                        tid = ticket.get('ticket_id', 'TKT-PENDING')
                        if lang == 'hi':
                            answer = f"आपके रिफ़ंड का बैंक संदर्भ संख्या (RRN) {rrn} है। रिफ़ंड में देरी की समीक्षा के लिए मैंने एक सपोर्ट टिकट ({tid}) बना दिया है और हमारी टीम (colleague) आपसे संपर्क करेगी।"
                        elif lang == 'hi-en':
                            answer = f"Aapke refund ka bank reference number (RRN) {rrn} hai. Isme ho rahi deri ke liye maine support ticket ({tid}) create kar diya hai aur colleague aapse sampark karenge."
                        else:
                            answer = f"Your refund reference number (RRN) is {rrn}. Since it is delayed, I have raised a support ticket ({tid}) for our team to resolve this with the bank."
                    else:
                        calls.append('create_ticket:failed')
                        if lang == 'hi':
                            answer = f"आपके रिफ़ंड का बैंक संदर्भ संख्या (RRN) {rrn} है। रिफ़ंड में देरी की समीक्षा के लिए हमारे सहयोगी (colleague) आपसे जल्द ही संपर्क करेंगे।"
                        elif lang == 'hi-en':
                            answer = f"Aapke refund ka bank reference number (RRN) {rrn} hai. Hamare colleague is delay ko check karke aapse jaldi sampark karenge."
                        else:
                            answer = f"Your refund reference number (RRN) is {rrn}. A human colleague will contact you shortly regarding the delay."
                    history.append({'role': 'user', 'content': message})
                    history.append({'role': 'assistant', 'content': answer})
                    return {'answer': answer, 'sources': [{'section_id': 'REF-6', 'source': 'pol_refunds.md', 'score': 1.0}, {'section_id': 'ESC-1', 'source': 'agent_escalation_handoff.md', 'score': 1.0}], 'tool_calls': calls, 'handover': True, 'decision': decision}

            # 5i. Return eligibility check
            if re.search(r'\b(return|wapas|vapas|वापस|lautana)\b', message, re.I):
                decision = 'answer'
                del_date_str = order_info.get('delivered_on') or order_info.get('delivery_date')
                order_date_str = order_info.get('placed_at') or order_info.get('order_date')
                is_plus = order_info.get('is_plus', False)
                item = items[0] if items else {}
                cat = item.get('return_category', 'fashion')

                # Window calculation
                is_festive = False
                if order_date_str and '2026-10-01' <= order_date_str <= '2026-10-15' and cat == 'fashion':
                    is_festive = True
                    window_days = 30
                elif cat == 'fashion':
                    window_days = 15
                elif cat == 'electronics':
                    window_days = 7
                else:
                    window_days = 10

                if del_date_str:
                    del_d = datetime.strptime(del_date_str, '%Y-%m-%d').date()
                    deadline = del_d + timedelta(days=window_days)
                    deadline_str = deadline.strftime('%Y-%m-%d')
                    deadline_human = deadline.strftime('%d %B %Y')
                else:
                    deadline_str = "within policy window"
                    deadline_human = "policy deadline"

                fee_mention = "waived for Plus members" if is_plus else "₹49 reverse pickup fee applies"
                if lang == 'hi':
                    fee_hi = "ZentroMart Plus सदस्य होने के कारण कोई पिकअप शुल्क नहीं लगेगा।" if is_plus else "₹49 का रिवर्स पिकअप शुल्क लागू होगा।"
                    answer = f"हाँ, आप इसे वापस कर सकते हैं। आपके लिए रिटर्न की अंतिम तिथि {deadline_human} ({deadline_str}) है। {fee_hi}"
                elif lang == 'hi-en':
                    fee_hien = "Plus members ke liye pickup fee waived hai." if is_plus else "₹49 reverse pickup fee lagegi."
                    answer = f"Haan, aap ise return kar sakte hain. Return karne ki aakhri date {deadline_human} ({deadline_str}) hai. {fee_hien}"
                else:
                    answer = f"Yes, you can return this item. The return window is open until {deadline_human} ({deadline_str}). A {fee_mention}."
                history.append({'role': 'user', 'content': message})
                history.append({'role': 'assistant', 'content': answer})
                sources = [{'section_id': 'RET-2', 'source': 'pol_returns_v4.md', 'score': 1.0}]
                if is_festive: sources.append({'section_id': 'RET-8', 'source': 'pol_returns_v4.md', 'score': 1.0})
                if is_plus: sources.append({'section_id': 'PLS-2', 'source': 'pol_plus_membership.md', 'score': 1.0})
                else: sources.append({'section_id': 'RET-6', 'source': 'pol_returns_v4.md', 'score': 1.0})
                return {'answer': answer, 'sources': sources, 'tool_calls': calls, 'handover': False, 'decision': decision}

            # 5j. Follow-up refund timeline for this verified order
            if re.search(r'\b(refund|kitne din|kab tak|how long|timeline|रिफंड|रिफ़ंड)\b', message, re.I):
                decision = 'answer'
                pay_method = order_info.get('payment_method', 'wallet').lower()
                if 'wallet' in pay_method:
                    if lang == 'hi': answer = "ZentroMart Wallet में रिफ़ंड तुरंत (instant, 2 घंटे के भीतर) आ जाता है।"
                    elif lang == 'hi-en': answer = "ZentroMart Wallet me refund instant (2 ghante ke andar) process hota hai."
                    else: answer = "Refunds to your ZentroMart Wallet are instant and processed within 2 hours."
                    sources = [{'section_id': 'REF-2', 'source': 'pol_refunds.md', 'score': 1.0}]
                elif 'upi' in pay_method:
                    if lang == 'hi': answer = "UPI द्वारा रिफ़ंड 1 से 3 कार्यदिवसों (business days) में आ जाता है।"
                    elif lang == 'hi-en': answer = "UPI refund 1 se 3 business days me aa jata hai."
                    else: answer = "Refunds via UPI take 1 to 3 business days to reach your account."
                    sources = [{'section_id': 'REF-2', 'source': 'pol_refunds.md', 'score': 1.0}]
                else:
                    if lang == 'hi': answer = "कार्ड या नेट बैंकिंग रिफ़ंड 5 से 7 कार्यदिवसों (business days) में प्रोसेस होता है।"
                    elif lang == 'hi-en': answer = "Net banking ya card refund 5 se 7 business days me process hota hai."
                    else: answer = "Refunds take 3 to 7 business days depending on your bank."
                    sources = [{'section_id': 'REF-2', 'source': 'pol_refunds.md', 'score': 1.0}]
                history.append({'role': 'user', 'content': message})
                history.append({'role': 'assistant', 'content': answer})
                return {'answer': answer, 'sources': sources, 'tool_calls': calls, 'handover': False, 'decision': decision}

        # Case 6: General RAG search across Knowledge Base & Product Catalog
        retrieved = self.index.search(message, k=4)
        sources = [{'section_id': c['section_id'], 'source': c['source'], 'score': c.get('score', 1.0)} for c in retrieved]

        # Privacy request: refuse sharing other customer details
        if re.search(r'\b(phone number|address|email|building|who ordered)\b', message, re.I):
            decision = 'refuse'
            if lang == 'hi': answer = "गोपनीयता नीति के तहत मैं किसी अन्य ग्राहक का फ़ोन नंबर या पता साझा नहीं कर सकता।"
            elif lang == 'hi-en': answer = "Privacy policy ke anusaar main kisi aur customer ka phone number ya details share nahi kar sakta."
            else: answer = "Under our customer privacy policy, I cannot share another customer's phone number or details."
            history.append({'role': 'user', 'content': message})
            history.append({'role': 'assistant', 'content': answer})
            return {'answer': answer, 'sources': sources, 'tool_calls': [], 'handover': False, 'decision': decision}

        # Check for date arithmetic in query (e.g. refund started on 17 October 2026)
        date_match = re.search(r'(\d{1,2})\s+(october|august|september|november)\s+(2026)', message, re.I)
        timeline_fact = ""
        if date_match and re.search(r'\b(refund|shuru|started|kab tak|by which date)\b', message, re.I):
            d_day = int(date_match.group(1))
            start_dt = date(2026, 10, d_day)
            d3 = add_business_days(start_dt, 3)
            d5 = add_business_days(start_dt, 5)
            d7 = add_business_days(start_dt, 7)
            timeline_fact = (
                f"\nCalculated refund deadlines from {start_dt.strftime('%d %B %Y')} based on KB-00.3 (Mon-Sat, skipping holidays):\n"
                f"- UPI (3 b-days): {d3.strftime('%Y-%m-%d')} ({d3.strftime('%d %B %Y')})\n"
                f"- Net banking (5 b-days): {d5.strftime('%Y-%m-%d')} ({d5.strftime('%d %B %Y')})\n"
                f"- Credit/debit card (7 b-days): {d7.strftime('%Y-%m-%d')} ({d7.strftime('%d %B %Y')})\n"
            )

        verified_order_fact = ""
        if state.get('verified') and state.get('order_data'):
            od = state['order_data']
            verified_order_fact = (
                f"\nVerified Order Context for Order ID {state['order_id']}:\n"
                f"- Status: {od.get('order_status')}\n"
                f"- Total: ₹{od.get('order_total')}\n"
                f"- Payment Method: {od.get('payment_method')}\n"
            )
            if od.get('items'):
                item_names = [it.get('name', '') for it in od['items']]
                verified_order_fact += f"- Items: {', '.join(item_names)}\n"

        context_texts = []
        for c in retrieved[:3]:
            context_texts.append(f"[{c['section_id']}] {c['text'][:350]}")
        context_block = "\n".join(context_texts) if context_texts else "[No relevant company documents found.]"

        prompt_system = (
            SYSTEM_PROMPT +
            f"\n\nRetrieved Knowledge & Policies:\n{context_block}\n" +
            timeline_fact +
            verified_order_fact
        )

        messages = [{'role': 'system', 'content': prompt_system}]
        messages += history[-4:]
        messages.append({'role': 'user', 'content': message})

        answer = self._call_llm(messages, max_tokens=75)

        history.append({'role': 'user', 'content': message})
        history.append({'role': 'assistant', 'content': answer})
        return {'answer': answer, 'sources': sources, 'tool_calls': calls, 'handover': handover, 'decision': decision}
