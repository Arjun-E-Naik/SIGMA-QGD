# -*- coding: utf-8 -*-
"""
Measure burstiness (sentence length distribution) and check sentence starters.
"""

import re
import numpy as np

with open('sigma_qgd_paper_v1.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Strip math environments, tables, algorithm, and comments
cleaned = re.sub(r'%.*', '', text)
cleaned = re.sub(r'\\begin\{(?:equation|align|tabular|algorithm)\}.*?\\end\{(?:equation|align|tabular|algorithm)\}', ' ', cleaned, flags=re.DOTALL)
cleaned = re.sub(r'\$\$?.*?\$\$?', ' ', cleaned)
cleaned = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?', ' ', cleaned)
cleaned = re.sub(r'[\{\}\[\]\(\)]', ' ', cleaned)

# Split into sentences
raw_sentences = re.split(r'(?<=[.!?])\s+', cleaned)
sentences = [s.strip() for s in raw_sentences if len(s.strip().split()) >= 4]

lengths = [len(s.split()) for s in sentences]

print(f"Total sentences analyzed: {len(sentences)}")
print(f"Mean sentence length: {np.mean(lengths):.2f} words")
print(f"Std deviation (burstiness): {np.std(lengths):.2f} words")
print(f"Min length: {np.min(lengths)}, Max length: {np.max(lengths)}")
print(f"Median length: {np.median(lengths):.1f}")
print(f"IQR (Q75 - Q25): {np.percentile(lengths, 75) - np.percentile(lengths, 25):.1f}")

# Distribution bins
short = sum(1 for l in lengths if l <= 12)
medium = sum(1 for l in lengths if 13 <= l <= 22)
long = sum(1 for l in lengths if l >= 23)

print(f"Short sentences (<=12 words): {short} ({short/len(lengths)*100:.1f}%)")
print(f"Medium sentences (13-22 words): {medium} ({medium/len(lengths)*100:.1f}%)")
print(f"Long sentences (>=23 words): {long} ({long/len(lengths)*100:.1f}%)")
