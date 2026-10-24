# -*- coding: utf-8 -*-
import unicodedata

with open("sigma_qgd_paper_v1.tex", "r", encoding="utf-8") as f:
    text = f.read()

non_ascii = set(c for c in text if ord(c) > 127)
print(f"Total unique non-ASCII characters: {len(non_ascii)}")
for c in sorted(non_ascii):
    count = text.count(c)
    name = unicodedata.name(c, "UNKNOWN")
    print(f"Char: {repr(c)} | Hex: {hex(ord(c))} | Count: {count:4d} | Name: {name}")
