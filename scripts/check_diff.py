# -*- coding: utf-8 -*-
with open('sigma_qgd_paper_draft.tex', 'r', encoding='utf-8') as f:
    draft = f.read()

pos_d = 10630 + 3480
print(repr(draft[pos_d:pos_d+100]))
