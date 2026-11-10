import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling_core.types.doc import TextItem, TableItem

DATA_DIR = Path(__file__).resolve().parent
PDF_DIR = DATA_DIR / "PDFs"
OUTPUT_DIR = DATA_DIR / "databases_qwen" / "chunked_reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text):
    return '\n'.join(' '.join(line.split()) for line in text.splitlines() if line.strip())


def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        chunk = text[start:start+chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def parse_pdf():
    for pdf in PDF_DIR.glob("*.pdf"):
        output_path = OUTPUT_DIR / f"{pdf.stem}.json"
        if output_path.exists():
            continue
        converter = DocumentConverter()
        result = converter.convert(pdf)
        document = result.document
        # document.save_as_json(f"{pdf.stem}docling_debug.json")
        print("Docling parsing completed.")
        
        pages = {}

        for item, level in document.iterate_items():
            if not item.prov:
                continue

            page_number = item.prov[0].page_no

            if isinstance(item, TextItem):
                text = item.text

            elif isinstance(item, TableItem):
                text = item.export_to_markdown(doc=document)

            else:
                continue

            text = clean_text(text)

            if text:
                pages.setdefault(page_number, []).append(text)

        chunks = []
        for page_number in sorted(pages):
            page_text = "\n\n".join(pages[page_number])
            for text in chunk_text(page_text):
                chunks.append({
                    "id":len(chunks),
                    "page": page_number,
                    "text": text,
                    "type": "content"
                })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "metainfoo":{
                    "sha1": pdf.stem,
                    "pages_amount": len(document.pages)
                },
                "content":{
                    "chunks": chunks
                }
            },
            f,
            ensure_ascii = False,
            indent=2)

        print(f"Saved {len(chunks)} chunks: {output_path}")
        # break


if __name__ == "__main__":
    parse_pdf()
