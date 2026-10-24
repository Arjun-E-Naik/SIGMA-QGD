# -*- coding: utf-8 -*-
with open('sigma_qgd_paper_draft.tex', 'r', encoding='utf-8') as f:
    draft = f.read()

import importlib.util
spec = importlib.util.spec_from_file_location("builder", "scripts/build_clean_v1.py")
builder = importlib.util.module_from_spec(spec)
# we don't execute because of exit(1), let's inspect reps from file
reps = []
with open("scripts/build_clean_v1.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Let's find why each target is failing
tests = [
    ("Sec 2.2 Question", "A natural and fundamental question arises"),
    ("Sec 3.3 after eq", "where $c > 0$ is the preconditioner multiplier"),
    ("Remark 1 end", "This sign inversion occurs because the opposing coordinate"),
    ("Remark 2 intro", "Three structural aspects of Proposition"),
    ("Remark 2 pt2", "demonstrates that trajectory variance scales with the square"),
    ("Empirical Validity paragraph", "Empirical Validity and Dynamical Self-Damping"),
    ("Sec 5.4.1 TFIM", "Performance on the Commuting TFIM Chain"),
    ("Sec 5.4.2 XXZ", "Performance on the Frustrated Heisenberg XXZ Chain"),
    ("Sec 5.4.3 Baselines", "Comparison with Stochastic and Natural Gradient Baselines"),
    ("Sec 5.4.4 Pareto", "Circuit Pareto Frontier Analysis"),
    ("Sec 5.6 bullets", "graceful error decay as shot budgets increase"),
    ("Sec 6.2", "Ansatz Expressivity and Landscape Frustration"),
    ("Sec 6.3", "Limitations"),
    ("Sec 7 Conclusion", "In this work, we introduced")
]

for label, phrase in tests:
    pos = draft.find(phrase)
    print(f"{label}: found={pos != -1} at pos={pos}")
    if pos != -1:
        print("   Snippet in draft:", repr(draft[pos:pos+60]))
