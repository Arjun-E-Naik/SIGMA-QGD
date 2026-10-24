# -*- coding: utf-8 -*-
with open('sigma_qgd_paper_draft.tex', 'r', encoding='utf-8') as f:
    draft = f.read()

import re

banned_list = [
    "cornerstone",
    "notoriously challenging",
    "severe non-stationarity",
    "coexist with",
    "Crucially",
    "Remarkably",
    "Notably",
    "Importantly",
    "Indeed",
    "Consequently",
    "elucidates the origin of this disparity",
    "in stark contrast",
    "bridges this divide",
    "bridges the gap",
    "To rigorously assess",
    "To systematically isolate",
    "exhibits graceful error decay",
    "computational and memory parsimony",
    "Computational and Memory Parsimony",
    "warrant explicit clarification",
    "delineate the natural boundary",
    "catapulting parameters out of"
]

print("=== SEARCHING BANNED PHRASES IN DRAFT ===")
for b in banned_list:
    matches = list(re.finditer(re.escape(b), draft, re.IGNORECASE))
    print(f"'{b}': {len(matches)} matches")
    for m in matches:
        start = max(0, m.start() - 60)
        end = min(len(draft), m.end() + 60)
        snippet = draft[start:end].replace('\n', ' ')
        print(f"   [Pos {m.start()}]: ...{snippet}...")
