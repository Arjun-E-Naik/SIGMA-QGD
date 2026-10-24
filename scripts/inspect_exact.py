# -*- coding: utf-8 -*-
with open('sigma_qgd_paper_draft.tex', 'r', encoding='utf-8') as f:
    draft = f.read()

import re

# 1. Remark 1
pos = draft.find(r"\begin{remark}")
print("=== REMARK 1 ===")
print(repr(draft[pos:pos+1200]))

# 2. Remark 2
pos2 = draft.find(r"\begin{remark}", pos+1)
print("=== REMARK 2 ===")
print(repr(draft[pos2:pos2+1200]))

# 3. Sec 5.4.2
pos3 = draft.find(r"\subsubsection{Performance on the Frustrated Heisenberg XXZ Chain}")
print("=== SEC 5.4.2 ===")
print(repr(draft[pos3:pos3+1200]))

# 4. Sec 5.4.3
pos4 = draft.find(r"\subsubsection{Comparison with Stochastic and Natural Gradient Baselines}")
print("=== SEC 5.4.3 ===")
print(repr(draft[pos4:pos4+1200]))

# 5. Sec 6.3
pos5 = draft.find(r"\subsection{Limitations}")
print("=== SEC 6.3 ===")
print(repr(draft[pos5:pos5+1200]))

# 6. Conclusion
pos6 = draft.find(r"\section{Conclusion}")
print("=== CONCLUSION ===")
print(repr(draft[pos6:pos6+1200]))
