# -*- coding: utf-8 -*-
with open('sigma_qgd_paper_draft.tex', 'r', encoding='utf-8') as f:
    draft = f.read()

# Let's inspect the exact regions in draft:
# 1. Sec 2.2 Question
pos1 = draft.find("A natural and fundamental question arises")
end1 = draft.find(r"\subsection{Quasi-Newton", pos1)
print("=== Sec 2.2 Question in Draft ===")
print(repr(draft[pos1:end1]))

# 2. Empirical Validity paragraph
pos2 = draft.find(r"\paragraph{Empirical Validity and Dynamical Self-Damping")
end2 = draft.find(r"\begin{theorem}", pos2)
print("=== Empirical Validity in Draft ===")
print(repr(draft[pos2:end2]))

# 3. Sec 5.4.1 TFIM
pos3 = draft.find(r"\subsubsection{Performance on the Commuting TFIM Chain}")
end3 = draft.find(r"\subsubsection{Performance on the Frustrated Heisenberg XXZ Chain}", pos3)
print("=== Sec 5.4.1 TFIM in Draft ===")
print(repr(draft[pos3:end3]))
