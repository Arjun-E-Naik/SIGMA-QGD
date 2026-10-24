# -*- coding: utf-8 -*-
"""
Surgical generator for sigma_qgd_paper_v1.tex and sigama_qgd_paper_v1.tex
Takes sigma_qgd_paper_draft.tex as base and performs surgical humanization
while preserving all math, theoretical blocks, algorithm, tables, and figures.
"""

import re
import os
import shutil

DRAFT_PATH = r"C:\Users\arjun\Downloads\mysigma\sigma_qgd_paper_draft.tex"
TARGET_V1 = r"C:\Users\arjun\Downloads\mysigma\sigma_qgd_paper_v1.tex"
TARGET_V1_ALT = r"C:\Users\arjun\Downloads\mysigma\sigama_qgd_paper_v1.tex"

with open(DRAFT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Verify initial sanity of draft
assert r"\begin{document}" in text
assert r"\end{document}" in text

print(f"Original draft length: {len(text)} characters")
