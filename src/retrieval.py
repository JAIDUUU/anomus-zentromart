"""Section-aware BM25 retrieval index over ZentroMart Markdown docs and product catalog.

Preserves exact section IDs (e.g. KB-00.1, RET-2, RETv3-2, products.csv:SKU),
enforces document precedence (policy > agent_guide > help > faq > marketing),
handles Devanagari/Hindi/Hinglish synonym expansion, and separates product lookup
from policy retrieval so product chunks never crowd out essential policy sections.
"""
import csv, math, re
from pathlib import Path
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+|[ऀ-ॿ]+", re.I)

def tokens(text):
    return TOKEN_RE.findall(text.lower())

def parse_frontmatter(text):
    meta = {}
    if text.startswith('---'):
        parts = text.split('---', 2)
        if len(parts) > 2:
            for line in parts[1].splitlines():
                if ':' in line:
                    k, v = line.split(':', 1)
                    meta[k.strip()] = v.strip()
            return meta, parts[2]
    return meta, text

HINDI_SYNONYMS = {
    'वापस': 'return returns return_category window',
    'लौटाने': 'return returns window days',
    'लौटा': 'return',
    'खोल': 'opened unsealed condition seal beauty',
    'सील': 'seal condition unopened',
    'टूटा': 'damaged defective crack broken reporting 48 hours',
    'सामान': 'item product',
    'दिन': 'days window',
    'रिफंड': 'refund timeline payment',
    'रिफ़ंड': 'refund timeline payment',
    'शुरू': 'initiated timeline business days',
    'पैसे': 'refund money',
    'कैंसिल': 'cancel cancellation shipped door refuse',
    'रद्द': 'cancel cancellation',
    'शिप': 'shipped cancel refuse door transit',
    'दुकान': 'store physical offline location',
    'वारंटी': 'warranty assured brand months',
    'क्लेम': 'claim assured my orders app',
    'सहयोगी': 'colleague human handover ticket',
    'मदद': 'help support assistant',
}

def expand_query(q):
    expanded = q
    for k, v in HINDI_SYNONYMS.items():
        if k in q:
            expanded += ' ' + v
    # Hinglish domain keywords
    if re.search(r'\b(khol|kholi|khole)\b', q, re.I):
        expanded += ' opened condition seal beauty'
    if re.search(r'\b(vapas|wapas|lautane|lautana|lauta)\b', q, re.I):
        expanded += ' return returns return_category window days'
    if re.search(r'\b(toota|damage|crack|tuta|phuta)\b', q, re.I):
        expanded += ' damaged defective crack 48 hours priority resolution'
    if re.search(r'\b(shuru)\b', q, re.I):
        expanded += ' initiated timeline business days'
    if re.search(r'\b(parso)\b', q, re.I):
        expanded += ' delivery date return window days'
    if re.search(r'\b(wallet)\b', q, re.I):
        expanded += ' refund precedence instant 2 hours wallet'
    if re.search(r'\b(net banking|upi|credit card|debit card)\b', q, re.I):
        expanded += ' refund timeline business days'
    if re.search(r'\b(sale)\b', q, re.I):
        expanded += ' festive sale 30 days october'
    if re.search(r'\b(never placed|fraud|didn\'t place|unauthorised)\b', q, re.I):
        expanded += ' escalation handoff human colleague ticket'
    if re.search(r'\b(collection|pickup fee|torn)\b', q, re.I):
        expanded += ' reverse pickup fee 49 waived damaged'
    if re.search(r'\b(human|person|agent|insaan|colleague)\b', q, re.I):
        expanded += ' escalation human colleague ticket'
    return expanded

def load_chunks(kb_dir, data_dir=None):
    chunks = []
    kb_path = Path(kb_dir)
    for path in sorted(kb_path.glob('*.md')):
        raw = path.read_text(encoding='utf-8')
        meta, body = parse_frontmatter(raw)
        # Match case-insensitively to preserve IDs like RETv3-2, KB-00.1, etc.
        pieces = re.split(r'(?mi)(?=^#{1,4} \[[a-z0-9._-]+\])', body)
        for piece in pieces:
            piece = piece.strip()
            if not piece: continue
            match = re.search(r'(?i)^#{1,4} \[([^]]+)\]\s*(.*)', piece)
            if not match: continue
            status = meta.get('status', 'current')
            chunks.append({
                'source': path.name,
                'section_id': match.group(1),
                'title': match.group(2).strip(),
                'status': status,
                'effective_from': meta.get('effective_from', ''),
                'doc_type': meta.get('doc_type', ''),
                'text': piece
            })

    # Also load products from data/products.csv if available
    if data_dir is None:
        data_dir = kb_path.parent / 'data'
    prod_file = Path(data_dir) / 'products.csv'
    if prod_file.exists():
        with prod_file.open(encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sku = row['sku'].strip()
                title = row.get('title', '').strip()
                text = (
                    f"Product SKU {sku}: {title}. Brand: {row.get('brand', '')}. "
                    f"Category: {row.get('return_category', '')}. Price: {row.get('price_inr', '')} INR. "
                    f"Fulfilled by: {row.get('fulfilled_by', '')}. ZentroMart Assured: {row.get('assured', '')}. "
                    f"Warranty: {row.get('warranty_months', '')} months. Description: {row.get('description', '')}"
                )
                chunks.append({
                    'source': 'products.csv',
                    'section_id': f'products.csv:{sku}',
                    'title': title,
                    'status': 'current',
                    'effective_from': '',
                    'doc_type': 'product_catalog',
                    'text': text
                })
    return chunks

class BM25Index:
    """Hybrid Section & Product BM25 retrieval index."""
    DOC_TYPE_WEIGHTS = {
        'policy': 1.35,
        'agent_guide': 1.15,
        'help': 1.0,
        'faq': 0.8,
        'marketing': 0.6,
        'product_catalog': 1.2,
    }

    def __init__(self, chunks):
        self.chunks = chunks
        self.kb_chunks = [c for c in chunks if c.get('doc_type') != 'product_catalog']
        self.prod_chunks = [c for c in chunks if c.get('doc_type') == 'product_catalog']

        self.kb_docs = [tokens(c['text']) for c in self.kb_chunks]
        self.kb_lens = [len(d) for d in self.kb_docs]
        self.kb_avgdl = sum(self.kb_lens) / max(1, len(self.kb_lens))
        self.kb_df = Counter(w for d in self.kb_docs for w in set(d))
        self.kb_N = len(self.kb_docs)

    def _search_products(self, query, k=2):
        # 1. Exact SKU match
        sku_matches = re.findall(r'\b([A-Z]{3}-\d{4})\b', query, re.I)
        if sku_matches:
            found = []
            for s in sku_matches:
                for c in self.prod_chunks:
                    if s.lower() in c['section_id'].lower():
                        found.append(dict(c, score=100.0))
            if found:
                return found[:k]

        # 2. Overlap with product title tokens
        q_tokens = set(tokens(query))
        scored = []
        for c in self.prod_chunks:
            title_toks = set(tokens(c['title']))
            overlap = len(q_tokens & title_toks)
            if overlap >= 2:
                scored.append((overlap, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [dict(c, score=float(s)) for s, c in scored[:k]]

    def search(self, query, k=6):
        # Search relevant products (up to 2)
        prods = self._search_products(query, k=2)

        # Expand query for KB search
        eq = expand_query(query)
        q = tokens(eq)
        allows_superseded = bool(re.search(r'\b(superseded|earlier|previous|v3|before 1 august|july)\b', query, re.I))

        scored_kb = []
        for i, doc in enumerate(self.kb_docs):
            tf = Counter(doc)
            score = 0.0
            for term in q:
                if term not in tf: continue
                df = self.kb_df[term]
                idf = math.log(1 + (self.kb_N - df + 0.5) / (df + 0.5))
                freq = tf[term]
                denom = freq + 1.5 * (1 - 0.75 + 0.75 * self.kb_lens[i] / max(1, self.kb_avgdl))
                score += idf * (freq * 2.5) / denom

            c = self.kb_chunks[i]
            score *= self.DOC_TYPE_WEIGHTS.get(c.get('doc_type', ''), 1.0)

            # Superseded handling
            if c['status'].lower() == 'superseded' and not allows_superseded:
                score *= 0.05

            if score > 0:
                scored_kb.append((score, c))

        scored_kb.sort(key=lambda x: x[0], reverse=True)
        top_kb = [dict(c, score=round(s, 4)) for s, c in scored_kb[:k]]

        # Ensure essential anchors based on intent
        # Refund precedence & timeline
        if re.search(r'\b(wallet|net banking|upi|credit card|refund|paisa|paise)\b', query, re.I):
            for sec in ['REF-2', 'KB-00.1']:
                if not any(c['section_id'] == sec for c in top_kb):
                    item = next((c for c in self.kb_chunks if c['section_id'] == sec), None)
                    if item: top_kb.append(dict(item, score=1.0))
        if re.search(r'\b(refund|kab tak|by which date|12 october|14 october|17 october)\b', query, re.I):
            if not any(c['section_id'] == 'KB-00.3' for c in top_kb):
                kb03 = next((c for c in self.kb_chunks if c['section_id'] == 'KB-00.3'), None)
                if kb03: top_kb.append(dict(kb03, score=1.0))

        # Superseded electronics (delivered in July / before August)
        if re.search(r'\b(july|before 1 august|v3|earlier)\b', query, re.I):
            if not any(c['section_id'] == 'RETv3-2' for c in top_kb):
                retv3 = next((c for c in self.kb_chunks if c['section_id'] == 'RETv3-2'), None)
                if retv3: top_kb.append(dict(retv3, score=1.0))

        # Privacy / phone / address disclosure refusal
        if re.search(r'\b(phone number|address|email|building|who ordered)\b', query, re.I):
            for sec in ['IDV-4', 'ESC-4']:
                if not any(c['section_id'] == sec for c in top_kb):
                    item = next((c for c in self.kb_chunks if c['section_id'] == sec), None)
                    if item: top_kb.append(dict(item, score=1.0))

        # Products warranty / assured
        if prods and re.search(r'\b(assured|warranty)\b', query, re.I):
            for sec in ['WAR-2', 'WAR-1']:
                if not any(c['section_id'] == sec for c in top_kb):
                    item = next((c for c in self.kb_chunks if c['section_id'] == sec), None)
                    if item: top_kb.append(dict(item, score=1.0))

        # Products return window
        if prods and re.search(r'\b(return|lautane|send back|fit|window|days)\b', query, re.I):
            for sec in ['RET-2', 'RET-5', 'RET-4', 'RET-8']:
                if not any(c['section_id'] == sec for c in top_kb):
                    item = next((c for c in self.kb_chunks if c['section_id'] == sec), None)
                    if item: top_kb.append(dict(item, score=1.0))

        # Damaged / defective
        if re.search(r'\b(crack|damage|tuta|toota|torn)\b', query, re.I):
            for sec in ['DWM-1', 'DWM-2', 'DWM-3', 'ESC-1', 'RET-6']:
                if not any(c['section_id'] == sec for c in top_kb):
                    item = next((c for c in self.kb_chunks if c['section_id'] == sec), None)
                    if item: top_kb.append(dict(item, score=1.0))

        # Shipped cancellation
        if re.search(r'\b(ship|shipped|radd|cancel|stop)\b', query, re.I):
            if not any(c['section_id'] == 'CAN-2' for c in top_kb):
                can2 = next((c for c in self.kb_chunks if c['section_id'] == 'CAN-2'), None)
                if can2: top_kb.append(dict(can2, score=1.0))

        # Unauthorized order / human request
        if re.search(r'\b(never placed|unauthorised|fraud|human|person|insaan)\b', query, re.I):
            if not any(c['section_id'] == 'ESC-1' for c in top_kb):
                esc1 = next((c for c in self.kb_chunks if c['section_id'] == 'ESC-1'), None)
                if esc1: top_kb.append(dict(esc1, score=1.0))

        combined = prods + top_kb
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in combined:
            if c['section_id'] not in seen:
                seen.add(c['section_id'])
                unique.append(c)
        return unique
