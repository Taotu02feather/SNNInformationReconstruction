import os
from pathlib import Path

import fitz

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "converted_texts"
OUTPUT_DIR.mkdir(exist_ok=True)

pdf_paths = sorted([p for p in BASE_DIR.rglob("*.pdf")])

if not pdf_paths:
    raise SystemExit("No PDF files found under Algorithm/.")

for pdf_path in pdf_paths:
    rel_path = pdf_path.relative_to(BASE_DIR)
    txt_path = OUTPUT_DIR / rel_path.with_suffix(".txt")
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing {rel_path} -> {txt_path.relative_to(BASE_DIR)}")
    doc = fitz.open(pdf_path)
    text_parts = []
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        text_parts.append(f"=== PAGE {page_num} ===\n")
        text_parts.append(page_text)
        text_parts.append("\n")

    content = "".join(text_parts)
    content = content.replace("\r\n", "\n")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"# Converted from: {pdf_path.name}\n")
        f.write(f"# Original path: {rel_path.as_posix()}\n")
        f.write("\n")
        f.write(content)

print(f"\nConverted {len(pdf_paths)} PDF files to {OUTPUT_DIR.relative_to(BASE_DIR)}")
