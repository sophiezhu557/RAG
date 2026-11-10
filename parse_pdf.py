import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer, ChunkingSerializerProvider
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer, MarkdownParams

DATA_DIR = Path(__file__).resolve().parent
PDF_DIR = DATA_DIR / "PDFs"
OUTPUT_DIR = DATA_DIR / "databases_qwen" / "chunked_reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class MDTableSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(compact_tables=True),
        ) 


def parse_pdf():
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B"),
        max_tokens=300
    ) 
    chunker = HybridChunker(
                tokenizer=tokenizer,
                merge_peers=True,
                repeat_table_header=True,
                serializer_provider = MDTableSerializerProvider()
            ) 
    # 当前hybrid chunker不支持overlap tokens

    converter = DocumentConverter()
    for pdf in PDF_DIR.glob("*.pdf"):
        output_path = OUTPUT_DIR / f"{pdf.stem}.json"

        if output_path.exists():
            continue

        result = converter.convert(pdf)
        document = result.document
        # document.save_as_json(f"{pdf.stem}docling_debug.json")
        print("Docling parsing completed.")

        chunks = []
        
        for i, chunk in enumerate(list(chunker.chunk(dl_doc=document))):
            page_numbers = sorted({
                prov.page_no 
                for item in chunk.meta.doc_items
                for prov in (item.prov or [])
            })

            source_types = [item.label.value for item in chunk.meta.doc_items]
            chunks.append(
                {
                    "id": i,
                    "page": page_numbers[0] if page_numbers else None,
                    "headings": chunk.meta.headings or [],
                    "type": "table" if "table" in source_types else "content",
                    "text": chunker.contextualize(chunk)
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "metainfo":{
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
