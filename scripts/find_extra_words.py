# -*- coding: utf-8 -*-
with open('sigma_qgd_paper_draft.tex', 'r', encoding='utf-8') as f:
    draft = f.read()

import re

extra_words = [
    "delve", "harness", "pivotal", "testament", "tapestry",
    "substantiated", "furthermore", "moreover", "notably", "importantly",
    "paramount", "underscores", "interplay"
]

for w in extra_words:
    matches = list(re.finditer(r'\b' + re.escape(w) + r'\b', draft, re.IGNORECASE))
    print(f"'{w}': {len(matches)} matches")
    for m in matches[:3]:
        start = max(0, m.start() - 30)
        end = min(len(draft), m.end() + 50)
        snippet = draft[start:end].replace('\n', ' ')
        print(f"   [Pos {m.start()}]: ...{snippet}...")
