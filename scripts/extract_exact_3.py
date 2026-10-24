# -*- coding: utf-8 -*-
with open('sigma_qgd_paper_draft.tex', 'r', encoding='utf-8') as f:
    draft = f.read()

# Let's extract the exact snippets from draft
pos1 = draft.find("A natural and fundamental question arises")
end1 = draft.find(r"\subsection{Quasi-Newton", pos1)
exact_s2_2_q = draft[pos1:end1]

pos2 = draft.find(r"\paragraph{Empirical Validity and Dynamical Self-Damping")
end2 = draft.find(r"\begin{theorem}", pos2)
exact_dyn_para = draft[pos2:end2]

pos3 = draft.find(r"\subsubsection{Performance on the Commuting TFIM Chain}")
end3 = draft.find(r"\subsubsection{Performance on the Frustrated Heisenberg XXZ Chain}", pos3)
exact_s5_4_1 = draft[pos3:end3]

print("Extracted exact snippets successfully!")
print("s2_2_q len:", len(exact_s2_2_q))
print("dyn_para len:", len(exact_dyn_para))
print("s5_4_1 len:", len(exact_s5_4_1))
