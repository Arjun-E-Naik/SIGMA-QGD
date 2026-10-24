# -*- coding: utf-8 -*-
with open('sigma_qgd_paper_draft.tex', 'r', encoding='utf-8') as f:
    draft = f.read()

import re

openers = [
    "By extracting",
    "Rather than querying",
    "While simultaneous perturbation methods",
    "Moving to n=12 qubits",
    "Moving to $n=12$"
]

for op in openers:
    matches = list(re.finditer(re.escape(op), draft, re.IGNORECASE))
    print(f"'{op}': {len(matches)} matches")
    for m in matches:
        start = max(0, m.start() - 30)
        end = min(len(draft), m.end() + 60)
        snippet = draft[start:end].replace('\n', ' ')
        print(f"   [Pos {m.start()}]: ...{snippet}...")
