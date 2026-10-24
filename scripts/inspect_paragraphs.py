# -*- coding: utf-8 -*-
with open('sigma_qgd_paper_draft.tex', 'r', encoding='utf-8') as f:
    draft = f.read()

import re

# Let's inspect paragraphs around key sections
def show_region(title, start_pattern, end_pattern):
    m_s = re.search(start_pattern, draft)
    m_e = re.search(end_pattern, draft)
    if not m_s or not m_e:
        print(f"FAILED to find region: {title}")
        return
    s = m_s.start()
    e = m_e.start()
    print(f"=== {title} [Pos {s}:{e}] ===")
    print(draft[s:e])
    print("-" * 50)

# 1. Pos 74000 - 77000 (XXZ results)
show_region("Sec 5.4.2 XXZ", r"\\subsubsection\{Performance on the Frustrated Heisenberg XXZ Chain\}", r"\\subsubsection\{Comparison with Stochastic")
show_region("Sec 5.4.3 Baselines", r"\\subsubsection\{Comparison with Stochastic and Natural Gradient Baselines\}", r"\\subsubsection\{Circuit Pareto")
show_region("Remark 1", r"\\begin\{remark\}\[Aggregate Alignment", r"\\begin\{lemma\}")
show_region("Remark 2", r"\\begin\{remark\}\[Dynamical Variance Scaling", r"\\begin\{proposition\}\[Uniform Local")
show_region("Sec 6.2 & 6.3", r"\\subsection\{Ansatz Expressivity and Landscape Frustration\}", r"\\section\{Conclusion\}")
show_region("Conclusion", r"\\section\{Conclusion\}", r"\\section\*\{Code and Data Availability\}")
