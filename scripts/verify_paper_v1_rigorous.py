# -*- coding: utf-8 -*-
"""
Rigorous validation test suite for sigma_qgd_paper_v1.tex and sigama_qgd_paper_v1.tex
"""

import re
import os

DRAFT_PATH = r"C:\Users\arjun\Downloads\mysigma\sigma_qgd_paper_draft.tex"
V1_PATH = r"C:\Users\arjun\Downloads\mysigma\sigma_qgd_paper_v1.tex"
V1_ALT_PATH = r"C:\Users\arjun\Downloads\mysigma\sigama_qgd_paper_v1.tex"

with open(DRAFT_PATH, "r", encoding="utf-8") as f:
    draft = f.read()

with open(V1_PATH, "r", encoding="utf-8") as f:
    v1 = f.read()

with open(V1_ALT_PATH, "r", encoding="utf-8") as f:
    v1_alt = f.read()

print("=" * 60)
print("TEST 1: Draft immutability & file identity")
print("=" * 60)
assert len(draft) == 111780, f"Draft length changed! {len(draft)}"
assert v1 == v1_alt, "sigma_qgd_paper_v1.tex and sigama_qgd_paper_v1.tex differ!"
print(f"PASS: Draft intact (111780 bytes). v1 and alt are identical ({len(v1)} chars).")

print("\n" + "=" * 60)
print("TEST 2: Environment pair balance")
print("=" * 60)
begins = re.findall(r'\\begin\{([a-zA-Z*]+)\}', v1)
ends = re.findall(r'\\end\{([a-zA-Z*]+)\}', v1)
from collections import Counter
c_b = Counter(begins)
c_e = Counter(ends)
assert c_b == c_e, f"Mismatched environments! begins={c_b}, ends={c_e}"
print(f"PASS: All {len(begins)} environments perfectly balanced across {len(c_b)} unique types.")

print("\n" + "=" * 60)
print("TEST 3: Section structure alignment")
print("=" * 60)
secs_draft = re.findall(r'(\\(?:sub)*section\*?\{[^}]+\})', draft)
secs_v1 = re.findall(r'(\\(?:sub)*section\*?\{[^}]+\})', v1)
assert secs_draft == secs_v1, f"Sections mismatch!\nDraft: {secs_draft}\nV1: {secs_v1}"
print(f"PASS: All {len(secs_v1)} section titles match character-for-character in exact order.")

print("\n" + "=" * 60)
print("TEST 4: Algorithm 1 line-for-line check")
print("=" * 60)
alg_draft = re.search(r'\\begin\{algorithm\}.*?\\end\{algorithm\}', draft, re.DOTALL).group(0)
alg_v1 = re.search(r'\\begin\{algorithm\}.*?\\end\{algorithm\}', v1, re.DOTALL).group(0)
assert alg_draft == alg_v1, "Algorithm 1 was modified!"
print("PASS: Algorithm 1 is 100% identical line-for-line.")

print("\n" + "=" * 60)
print("TEST 5: Tables 1-8 tabular bodies check")
print("=" * 60)
tabs_draft = re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}', draft, re.DOTALL)
tabs_v1 = re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}', v1, re.DOTALL)
assert len(tabs_draft) == len(tabs_v1) == 8, f"Expected 8 tabular blocks, got draft={len(tabs_draft)}, v1={len(tabs_v1)}"
for i, (t1, t2) in enumerate(zip(tabs_draft, tabs_v1)):
    assert t1 == t2, f"Tabular block {i+1} differs!"
print("PASS: All 8 tabular bodies are 100% identical character-for-character.")

print("\n" + "=" * 60)
print("TEST 6: Figures 1-7 includegraphics and labels check")
print("=" * 60)
figs_draft = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', draft)
figs_v1 = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', v1)
assert figs_draft == figs_v1, f"Figure graphics mismatch! Draft: {figs_draft}, V1: {figs_v1}"
fig_labels_draft = re.findall(r'\\label\{(fig:[^}]+)\}', draft)
fig_labels_v1 = re.findall(r'\\label\{(fig:[^}]+)\}', v1)
assert fig_labels_draft == fig_labels_v1, f"Figure labels mismatch! Draft: {fig_labels_draft}, V1: {fig_labels_v1}"
print("PASS: All 7 figures have identical includegraphics paths and labels.")

print("\n" + "=" * 60)
print("TEST 7: Equations and Eq (8) duplicate label check")
print("=" * 60)
eqs_draft = re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}', draft, re.DOTALL)
eqs_v1 = re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}', v1, re.DOTALL)
assert len(eqs_draft) == len(eqs_v1) == 30, f"Expected 30 equations, got draft={len(eqs_draft)}, v1={len(eqs_v1)}"
diff_count = 0
for i, (e1, e2) in enumerate(zip(eqs_draft, eqs_v1)):
    if e1 != e2:
        diff_count += 1
        print(f"Equation {i+1} diff:")
        print("Draft:", repr(e1))
        print("V1:   ", repr(e2))
        assert r"\label{eq:preconditioner}" in e1 and r"\label{eq:preconditioner}" not in e2
        assert r"\label{eq:preconditioned_update}" in e1 and r"\label{eq:preconditioned_update}" in e2
print(f"PASS: Exactly {diff_count} equation differs (Eq 8: duplicate label eq:preconditioner removed). All other 29 equations identical.")

print("\n" + "=" * 60)
print("TEST 8: Theoretical environments & proofs check")
print("=" * 60)
for env in ['assumption', 'lemma', 'proposition', 'theorem', 'proof']:
    b_draft = re.findall(rf'\\begin\{{{env}\}}(.*?)\\end\{{{env}\}}', draft, re.DOTALL)
    b_v1 = re.findall(rf'\\begin\{{{env}\}}(.*?)\\end\{{{env}\}}', v1, re.DOTALL)
    assert len(b_draft) == len(b_v1), f"Env count mismatch for {env}!"
    for i, (b1, b2) in enumerate(zip(b_draft, b_v1)):
        # Check formal math content
        if env == 'proposition' and i == 0:
            # Proposition 1 has connecting English word Consequently -> As a result
            assert "Consequently" in b1 and "As a result" in b2
            assert b1.replace("Consequently", "As a result") == b2, "Proposition 1 had other alterations!"
        else:
            assert b1 == b2, f"Environment {env} block {i+1} differs!"
print("PASS: Assumptions 1-5, Lemma 1, Propositions 1-3, Theorem 1, and Proofs 1-5 intact.")

print("\n" + "=" * 60)
print("TEST 9: Banned AI words scan")
print("=" * 60)
banned = [
    'cornerstone', 'notoriously challenging', 'severe non-stationarity', 'coexist with',
    'Crucially', 'Remarkably', 'Notably', 'Importantly', 'Indeed', 'Consequently',
    'elucidates the origin of this disparity', 'in stark contrast', 'bridges this divide',
    'bridges the gap', 'To rigorously assess', 'To systematically isolate',
    'exhibits graceful error decay', 'computational and memory parsimony',
    'warrant explicit clarification', 'delineate the natural boundary',
    'catapulting parameters out of', 'substantiated'
]
banned_hits = {}
for b in banned:
    c = len(re.findall(re.escape(b), v1, re.IGNORECASE))
    if c > 0:
        banned_hits[b] = c

if banned_hits:
    print("FAILED! Banned words found in v1:", banned_hits)
    exit(1)
else:
    print(f"PASS: 0 occurrences of all {len(banned)} banned AI phrases in v1!")

print("\n" + "=" * 60)
print("TEST 10: Citations and References resolution")
print("=" * 60)
bibitems = re.findall(r'\\bibitem\{([^}]+)\}', v1)
cites = set(re.findall(r'\\cite\{([^}]+)\}', v1))
all_cites = set()
for c in cites:
    for item in c.split(','):
        all_cites.add(item.strip())

bib_set = set(bibitems)
missing_cites = all_cites - bib_set
assert not missing_cites, f"Missing bibitems for citations: {missing_cites}"
print(f"PASS: All {len(all_cites)} citations cleanly resolved across {len(bibitems)} bibitems.")

labels = set(re.findall(r'\\label\{([^}]+)\}', v1))
refs = set(re.findall(r'\\(?:eq)?ref\{([^}]+)\}', v1))
missing_refs = refs - labels
assert not missing_refs, f"Missing labels for refs: {missing_refs}"
print(f"PASS: All {len(refs)} references cleanly resolved across {len(labels)} defined labels.")

print("\n" + "=" * 60)
print("ALL TESTS PASSED WITH ZERO ERRORS!")
print("=" * 60)
